● 1. Why your A100‑40GB run stalled                  
                                           
  The command picked the a100 profile, which is tuned for the 80 GB SKU. Two things happen on a 40 GB notebook box:
                                          
  (a) CPU/IO pipeline saturation, not GPU OOM. The a100 profile sets num_workers=12, prefetch_factor=8, persistent_workers=True. With your override --mv_groups 192  
  the per‑batch sample count is 192*9 = 1728, so the prefetch queues hold up to 12 * 8 * 1728 ≈ 165 K decoded JPEGs (~25 GB CPU RAM) and pin a comparable amount of
  /dev/shm for MosaicML's worker barriers. On Colab/Kaggle/single‑GPU‑40GB cloud images, /dev/shm is typically 64 MB–2 GB and host RAM is ~80 GB — the first batch   
  finishes, all workers then try to refill simultaneously, and the loader deadlocks. From the outside it looks like "first batch printed, hangs forever". This       
  matches what you observed.                                                                                                                                         
                                                                                                                                                                     
  (b) GPU peak headroom. Empirically the 80 GB box peaks at ~54 GB at 224 mv_groups → ~0.24 GB per group. 192 groups would land near 46 GB peak — already over the 40
   GB cap, so you'd also OOM around the time the second backward starts allocating its tape. Whichever fires first (CPU stall or GPU OOM), the symptom is the same:  
  stuck.                                                                                                                                                             
                                                                                                                                                                     
  Fix. Use the new profile I added at RayNet/hardware_profiles.py:                                                                                                   

  python -m RayNet.train \
    --mds_streaming --mds_train ../shards/train/ --mds_val ../shards/val/ \                                                                                          
    --core_backbone_weight_path ../repnext_m1_distill_300e.pth \                                                                                                     
    --profile a100_40gb \                                                                                                                                            
    --samples_per_subject 2000 \                                                                                                                                     
    --no_compile \                                                                                                                                                   
    --ckpt_bucket raynet-checkpoints                                                                                                                             
                                                                                                                                                                   
  a100_40gb: 128 mv_groups (1152 batch), num_workers=6, prefetch_factor=4, BF16, TF32, no compile, persistent workers. If it still stalls, drop --num_workers 4.     
                                                                                                                                                                   
  2. The A100 80 GB only fills 54 GB — that's expected                                                                                                               
                                                                                                                                                                   
  The previous comment in the file was wrong. With BF16 + grad checkpointing on Triple‑M1 (~18.7M params) the activation footprint is smaller than the 78 GB the     
  comment predicted. I updated the comment in hardware_profiles.py:a100. To trade headroom for throughput on 80 GB, run with --mv_groups 320 (2880 samples) — that 
  should land near 75 GB peak. Check the rank‑0 GPU mem line printed at the end of epoch 1 and step up if headroom > 5 GB.                                           
                                                                                                                                                                   
  3. Data‑leak guarantees on resume / fork / warmstart                                                                                                               
   
  The train/val partition is established once, at MDS conversion time, by convert_to_mds_chunked(subject_ids=...) (streaming/convert_to_mds.py:160). train.py then   
  takes two CLI URLs (--mds_train, --mds_val) and wires them through _create_mds_mv_loader (train.py:1708) into two completely separate StreamingGazeGeneDataset   
  instances (streaming/dataset.py:260 and :272). The local default partition for the disk loaders is subjects 1‑46 train / 47‑56 val (train.py:1678).                
                                                                                                                                                                   
  What this means for the three continuation modes:                                                                                                                  
   
  - --resume (train.py:1101) — same run_id, same --mds_train/--mds_val, model + optimizer + scheduler + scaler all loaded; epoch counter advances from               
  ckpt['epoch']+1. No new data path is introduced, so no leak.                                                                                                     
  - --fork_from (train.py:1171) — new run_id, but the same --mds_train/--mds_val URLs you pass on the CLI. Source run is read‑only (load_from_run at                 
  streaming/checkpoint.py:316). No leak.                                                                                                                             
  - --warmstart_from (train.py:1318) — new run_id, model‑weights only, optimizer/scheduler/epoch fresh. Same shard URLs. No leak.
                                                                                                                                                                     
  The single thing that could create a leak is the conversion step itself — if you ever rebuild ../shards/train/ and ../shards/val/ from overlapping subject lists.  
  The streaming + checkpoint code does not re‑establish the partition; it only ever consumes whatever the two URLs already hold. Practical hygiene:                  
  - Keep subject_start/--subject_end for train and val runs disjoint when you regenerate shards.                                                                     
  - Don't change the shard URLs between the source run and the resume/fork/warmstart run, or you'll silently train on a different partition than you validated       
  against.                                                                                                                                                    
  - One subtle non‑leak caveat: the dataloader's RNG state is not persisted across --resume. End‑of‑epoch is the only save point (train.py:1517), so on resume the   
  loader simply starts a fresh epoch — at worst this changes shuffle order, not the partition.                                                                     
                                                                                                                                                                     
  4. How multi‑view groups behave when multiview is False                                                                                                          
                                                                                                                                                                     
  This part of the codebase is intentional but has two independent switches that often get conflated. Quoting the header comment at train.py:71-93:                  
                                                                                                                                                                     
  - The MDS shard layout is always 9‑grouped. _build_multiview_order (streaming/convert_to_mds.py:137) sorts samples so all 9 cameras for one (subject, frame) are   
  consecutive. The streaming loader then runs with shuffle=False and batch_size = mv_groups * 9 (streaming/dataset.py:351-361), so every batch you ever consume — P1,
   P2, or P3 — is exactly mv_groups complete (subject, frame) groups in cam_id order.                                                                                
  - cfg['multiview'] only gates the auxiliary multiview_consistency_loss (gaze‑consistency + shape terms in multiview_loss.py). It does NOT control whether        
  CrossViewAttention runs.                                                                                                                                           
  - active_n_views controls CrossViewAttention. It is set in the per‑epoch loop at train.py:1475-1478:
  if phase == 2 or args.no_multiview:                                                                                                                                
      active_n_views = 1                                                                                                                                             
  else:                                                                                                                                                              
      active_n_views = 9                                                                                                                                             
  - When n_views == 1, CrossViewAttention.forward short‑circuits to identity (raynet_v5.py:619-621), so the 9‑grouped batch is processed as 9·G independent monocular
   samples.                                                                                                                                                          
   
  So phase by phase:                                                                                                                                                 
                                                                                                                                                                   
  ┌──────────┬──────────────────┬────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   
  │  Phase   │ cfg['multiview'] │ active_n_views │                                              What actually runs                                               │
  ├──────────┼──────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ 1 (1‑15) │ False            │ 9              │ CrossViewAttention IS active and trains. The 'gaze_only' freeze set explicitly leaves cross_view_attn and     │ 
  │          │                  │                │ camera_embedding trainable (train.py:209-211). Consistency loss off.                                          │
  ├──────────┼──────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ 2        │ False            │ 1              │ CrossViewAttention is bypassed. Camera embedding receives no gradient. Loader still produces 9‑grouped        │
  │ (16‑30)  │                  │                │ batches but they are flattened into 9·G monocular samples. Consistency loss off.                              │   
  ├──────────┼──────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
  │ 3        │ True             │ 9              │ CrossViewAttention active + consistency loss with a 5‑epoch ramp anchored to epoch 30 (train.py:595).         │   
  │ (31‑50)  │                  │                │                                                                                                               │   
  └──────────┴──────────────────┴────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                                                                                                     
  A few practical consequences:                                                                                                                                    
  - Validation uses the same active_n_views as training (train.py:1505-1509), so val gaze metrics in P1 and P3 are fused; P2 val is pure monocular. Don't compare
  absolute val_angular across phase boundaries.                                                                                                                      
  So phase by phase:

  ┌──────────┬──────────────────┬────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │  Phase   │ cfg['multiview'] │ active_n_views │                                              What actually runs                                               │
  ├──────────┼──────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 1 (1‑15) │ False            │ 9              │ CrossViewAttention IS active and trains. The 'gaze_only' freeze set explicitly leaves cross_view_attn and     │
  │          │                  │                │ camera_embedding trainable (train.py:209-211). Consistency loss off.                                          │
  ├──────────┼──────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 2        │ False            │ 1              │ CrossViewAttention is bypassed. Camera embedding receives no gradient. Loader still produces 9‑grouped        │
  │ (16‑30)  │                  │                │ batches but they are flattened into 9·G monocular samples. Consistency loss off.                              │
  ├──────────┼──────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 3        │ True             │ 9              │ CrossViewAttention active + consistency loss with a 5‑epoch ramp anchored to epoch 30 (train.py:595).         │
  │ (31‑50)  │                  │                │                                                                                                               │
  └──────────┴──────────────────┴────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  A few practical consequences:
  - Validation uses the same active_n_views as training (train.py:1505-1509), so val gaze metrics in P1 and P3 are fused; P2 val is pure monocular. Don't compare
  absolute val_angular across phase boundaries.
  - If you wanted Phase 1 to also be monocular (e.g. to ablate whether CVA training during P1 helps), you'd pass --no_multiview — that's what flips both
  cfg['multiview'] to False AND forces active_n_views=1.
  - --mv_groups 192 doesn't change whether fusion happens; it only sets how many (subject, frame) groups are in each batch.

  ---
  Files changed: RayNet/hardware_profiles.py (added a100_40gb, corrected the a100 comment).
