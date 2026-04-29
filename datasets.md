 ---                                                                                                      
  Benchmark Gold-Standard Datasets Report                                                                  
                                                                                                           
  Deep-Prot Studio — RPI, DTPI, PDI, PPI Modules                                                            
                                                                                                           
  ---                                                                                                      
  1. RPI — RNA-Protein Interaction                                                                         
                                  
  Canonical Benchmark Datasets
                                                                                                           
  ┌──────────────┬────────────────────────┬──────────────────┬─────────────────────────────────────────┐
  │   Dataset    │         Pairs          │      Source      │                  Notes                  │   
  ├──────────────┼────────────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ RPI369       │ 369                    │ PDB              │ Smallest; used for generalization tests │
  ├──────────────┼────────────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ RPI488       │ 488                    │ PDB              │ Balanced positives/negatives            │   
  ├──────────────┼────────────────────────┼──────────────────┼─────────────────────────────────────────┤   
  │ RPI1446      │ 1,446                  │ PDB + literature │ Medium scale                            │   
  ├──────────────┼────────────────────────┼──────────────────┼─────────────────────────────────────────┤   
  │ RPI1807      │ 1,807                  │ PDB              │ Widely cited                            │
  ├──────────────┼────────────────────────┼──────────────────┼─────────────────────────────────────────┤   
  │ RPI2241      │ 2,241                  │ Literature       │ Largest classic set                     │
  ├──────────────┼────────────────────────┼──────────────────┼─────────────────────────────────────────┤   
  │ NPInter v3.0 │ 491,416 interactions   │ CAS/Biophysics   │ lncRNA + protein, 188 tissues           │
  ├──────────────┼────────────────────────┼──────────────────┼─────────────────────────────────────────┤   
  │ NPInter v5.0 │ 2,596,695 interactions │ CAS/Biophysics   │ Adds RNA-DNA, SARS-CoV-2, scRNA-seq     │
  └──────────────┴────────────────────────┴──────────────────┴─────────────────────────────────────────┘   
                  
  Key Papers (cite these for your benchmarking)                                                            
                  
  1. ZHMolGraph — GNN + LLM for unknown RNA/protein pairs; AUROC 79.8%, AUPRC 82.0%                        
  DOI: 10.1038/s42003-025-07694-9 | Commun. Biology 2025
  2. RPI-GGCN — Gated GCN + Co-VAE; 97.27% on RPI369, validated on NPInter v3.0                            
  DOI: 10.1109/TNNLS.2024.3390935 | IEEE TNNLS 2025                                                        
  3. RPIFSE — CNN + ELM ensemble; 98.98% on RPI1807                                                        
  DOI: 10.1016/j.jtbi.2018.10.029 | J. Theor. Biol. 2018                                                   
  4. RPITER — CNN + SAE; AUC 0.985 on NPInter                                                              
  DOI: 10.3390/ijms20051070 | Int. J. Mol. Sci. 2019                                                       
  5. NPInter v5.0 (database paper) — authoritative source for interaction data                             
  DOI: 10.1093/nar/gkac1002 | Nucleic Acids Res. 2023                                                      
  6. NPInter v3.0 (database paper)                                                                         
  DOI: 10.1093/database/baw057 | Database 2016                                                             
                                                                                                           
  ▎ Recommendation for Deep-Prot Studio: Use RPI369, RPI488, RPI1807, and RPI2241 as your four-tier        
  ▎ benchmark. Obtain them from the original dataset releases or GitHub repos of RPIFSE/RPITER. NPInter    
  ▎ v3.0 is the standard independent validation set.                                                       
                  
  ---
  2. DTPI — Drug-Target Protein Interaction

  Canonical Benchmark Datasets

  ┌─────────────────────────┬──────────────────┬───────────┬───────────────────────────────┬──────────┐    
  │         Dataset         │      Drugs       │  Targets  │             Task              │  Metric  │
  ├─────────────────────────┼──────────────────┼───────────┼───────────────────────────────┼──────────┤    
  │ Davis                   │ 68 kinase        │ 442       │ Affinity (Kd)                 │ MSE, CI  │
  │                         │ inhibitors       │ kinases   │                               │          │
  ├─────────────────────────┼──────────────────┼───────────┼───────────────────────────────┼──────────┤    
  │ KIBA                    │ 2,111 compounds  │ 229       │ Bioactivity score (Ki/Kd/IC50 │ MSE, CI  │    
  │                         │                  │ targets   │  fused)                       │          │    
  ├─────────────────────────┼──────────────────┼───────────┼───────────────────────────────┼──────────┤    
  │ Metz                    │ ~178 drugs       │ 170       │ IC50 affinity                 │ MSE      │
  │                         │                  │ kinases   │                               │          │    
  ├─────────────────────────┼──────────────────┼───────────┼───────────────────────────────┼──────────┤
  │ Human                   │ ~3,400           │ ~1,600    │ Binary (interaction/no        │ AUC,     │    
  │ (BindingDB-derived)     │                  │           │ interaction)                  │ AUPRC    │
  └─────────────────────────┴──────────────────┴───────────┴───────────────────────────────┴──────────┘

  Key Papers (cite these for your benchmarking)

  1. DeepNC — GNN framework; evaluated on Davis, KIBA, Allergy datasets                                    
  DOI: 10.7717/peerj.13163 | PeerJ 2022
  2. TDGraphDTA — Transformer + diffusion graph; Davis, Metz, KIBA                                         
  DOI: 10.1016/j.compbiolchem.2023.107621 | Comput. Biol. Med. 2023                                        
  3. Mutual-DTPI — Mutual interaction features + Transformer; two benchmark datasets                        
  DOI: 10.3934/mbe.2023469 | Math. Biosci. Eng. 2023                                                       
  4. BERT-DTPI — Subsequence embedding + transfer learning; three benchmark datasets                        
  DOI: 10.1016/j.compbiolchem.2024.108058 | Comput. Biol. Chem. 2024                                       
  5. PIGLET — Graph transformer on knowledge graph; Human dataset with drug-based split                    
  DOI: 10.64898/2026.02.18.706530 | bioRxiv 2026                                                           
  6. MGDTPI — Cold-start meta-learning graph transformer; standard benchmark                                
  DOI: 10.1016/j.ymeth.2024.11.010 | Methods 2024                                                          
  7. ETransDTA — CNN + Transformer; Davis & KIBA (CI, MSE metrics)                                         
  DOI: 10.1142/S0219720023500300 | J. Bioinform. Comput. Biol. 2024                                        
                                                                                                           
  ▎ Recommendation for Deep-Prot Studio: Davis and KIBA are the universal standard — every DTPI paper       
  ▎ benchmarks against them. Davis is simpler (binary-style with a Kd threshold), KIBA is more complex.    
  ▎ Obtain them from the DeepDTA/GraphDTA GitHub repositories where they are already preprocessed.         
                  
  ---
  3. PDI — Protein-DNA Interaction (Transcription Factor Binding Sites)
                                                                                                           
  Canonical Benchmark Datasets
                                                                                                           
  ┌───────────────────┬─────────────────────────┬───────────────────────┬─────────────────────────────┐
  │      Dataset      │          Scale          │         Type          │            Notes            │
  ├───────────────────┼─────────────────────────┼───────────────────────┼─────────────────────────────┤
  │ ENCODE ChIP-seq   │ 127–690 TFs, multiple   │ In vivo TF binding    │ Gold-standard for TFBS      │
  │                   │ cell lines              │                       │ prediction                  │
  ├───────────────────┼─────────────────────────┼───────────────────────┼─────────────────────────────┤    
  │ JASPAR 2022       │ 1,800+ motifs           │ Curated TF PWMs       │ Cross-species, open-access  │
  ├───────────────────┼─────────────────────────┼───────────────────────┼─────────────────────────────┤    
  │ TRANSFAC          │ 7,000+ entries          │ Curated TF PWMs       │ Commercial; widely used as  │
  │                   │                         │                       │ reference                   │    
  ├───────────────────┼─────────────────────────┼───────────────────────┼─────────────────────────────┤
  │ SNP-SELEX         │ genome-wide             │ In vitro allelic      │ For variant-effect          │    
  │                   │                         │ binding               │ prediction                  │    
  ├───────────────────┼─────────────────────────┼───────────────────────┼─────────────────────────────┤
  │ ENCODE-DREAM      │ 31 TFs, 5 cell types    │ In vivo competition   │ Official challenge standard │    
  │ Challenge         │                         │ benchmark             │                             │    
  └───────────────────┴─────────────────────────┴───────────────────────┴─────────────────────────────┘
                                                                                                           
  Key Papers (cite these for your benchmarking)

  1. TFBS Prediction Benchmark — PWM vs SVM vs DL, ENCODE ChIP-seq across cell lines; direct benchmarking  
  paper
  DOI: 10.1093/bib/bbaf363 | Brief. Bioinform. 2025                                                        
  2. maxATAC — 127-TF model suite for ATAC-seq; curated benchmark dataset released publicly                
  DOI: 10.1371/journal.pcbi.1010863 | PLoS Comput. Biol. 2023                                              
  3. DeepARC — CNN-BiLSTM-Attention; ENCODE ChIP-seq on A549, GM12878, HepG2, H1-hESC, HeLa; avg AUC 0.908 
  DOI: 10.3389/fonc.2022.893520 | Front. Oncol. 2022                                                       
  4. TF activity benchmark — 1,541 human TFs; compares ChIP-seq vs curated vs in-silico regulons; key      
  reference for dataset authority                                                                          
  DOI: 10.1101/gr.240663.118 | Genome Res. 2019                                                            
  5. SNP effects on TF-DNA binding — 14 models evaluated; SNP-SELEX + ASB data; comprehensive comparison   
  DOI: 10.1093/bib/bbae110 | Brief. Bioinform. 2024                                                        
                                                                                                           
  ▎ Recommendation for Deep-Prot Studio: Use the ENCODE DREAM Challenge data (publicly available from the  
  ▎ challenge website) as your core PDI benchmark. JASPAR 2022 (https://jaspar.elixir.no) provides the     
  ▎ motif reference. For sequence-based PDI, the 690 human ChIP-seq datasets from ENCODE are the de facto  
  ▎ Recommendation for Deep-Prot Studio: Use the ENCODE DREAM Challenge data (publicly available from the challenge website) as your
  ▎ core PDI benchmark. JASPAR 2022 (https://jaspar.elixir.no) provides the motif reference. For sequence-based PDI, the 690 human
  ▎ ChIP-seq datasets from ENCODE are the de facto gold standard.

  ---

  ┌────────────┬─────────────────────────────┬───────────────┬───────────────────────────────────────┐
  │  Dataset   │            Scale            │    Species    │                 Notes                 │
  ├────────────┼─────────────────────────────┼───────────────┼───────────────────────────────────────┤
  │ STRING v12 │ 67M+ PPIs, 14,094 organisms │ Multi-species │ Scored; confidence-tiered             │
  ├────────────┼─────────────────────────────┼───────────────┼───────────────────────────────────────┤
  │ SHS27k     │ 27,182 pairs                │ Human         │ Subset of STRING; balanced ML dataset │
  ├────────────┼─────────────────────────────┼───────────────┼───────────────────────────────────────┤
  │ SHS148k    │ 148,323 pairs               │ Human         │ Larger STRING subset                  │
  ├────────────┼─────────────────────────────┼───────────────┼───────────────────────────────────────┤
  │ DIP        │ ~80,000 interactions        │ Multi         │ Experimental; curated                 │
  ├────────────┼─────────────────────────────┼───────────────┼───────────────────────────────────────┤
  │ HPRD       │ ~39,000                     │ Human         │ Manually curated from literature      │
  ├────────────┼─────────────────────────────┼───────────────┼───────────────────────────────────────┤
  │ P2P        │ 6,886 complexes             │ Multi         │ For binding free energy prediction    │
  └────────────┴─────────────────────────────┴───────────────┴───────────────────────────────────────┘

  Key Papers (cite these for your benchmarking)

  1. STRING v12 — The primary reference database for PPI; used by virtually all DL PPI benchmarks
  DOI: 10.1093/nar/gkac1000 | Nucleic Acids Res. 2023
  2. PLNet — Persistent Laplacian NN for PPI binding free energy; P2P benchmark (6,886 complexes)
  DOI: 10.1002/pro.70377 | Protein Sci. 2025
  3. PHI prediction with GCN — Virus-human PPI; uses benchmark PHI dataset; 3–23% better AUC than baselines
  DOI: 10.1016/j.compbiolchem.2022.107755 | Comput. Biol. Chem. 2022

  ▎ Recommendation for Deep-Prot Studio: Use SHS27k and SHS148k (derived from STRING, available on GitHub from the PIPR/SPRINT papers) as
  ▎ primary benchmarks. These are the most used in sequence-based PPI DL. STRING v12 itself is the authoritative source for downloading
  ▎ interaction pairs with confidence scores.

  ---
  Summary Table

  ┌────────┬────────────────────────────────────────────────────┬───────────────────────────┐
  │ Module │                  Primary Datasets                  │      Primary Metrics      │
  ├────────┼────────────────────────────────────────────────────┼───────────────────────────┤
  │ RPI    │ RPI369, RPI488, RPI1807, RPI2241, NPInter v3.0     │ Accuracy, AUC, AUPRC      │
  ├────────┼────────────────────────────────────────────────────┼───────────────────────────┤
  │ DTPI    │ Davis, KIBA, Human (BindingDB)                     │ MSE, CI, AUROC, AUPRC     │
  ├────────┼────────────────────────────────────────────────────┼───────────────────────────┤
  │ PDI    │ ENCODE ChIP-seq (127 TFs), JASPAR, DREAM Challenge │ AUC, AUPR, F1             │
  ├────────┼────────────────────────────────────────────────────┼───────────────────────────┤
  │ PPI    │ SHS27k, SHS148k (STRING-derived), P2P              │ Accuracy, AUC, AUPRC, MCC │
  └────────┴────────────────────────────────────────────────────┴───────────────────────────┘

  All data above is based on articles retrieved from PubMed. All DOI links point to the original publications. Let me know if you want me
  to dig deeper into any specific module or find the data download links for any of these datasets.