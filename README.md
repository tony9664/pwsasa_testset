# pwsasa_testset

Testset for pwsasa project

Systems:
1. Arg dipeptide, 36 atoms > 32 (with possible warp issues)
2. Gly dipeptide, 19 atoms < 32

FF:
ff19SB

Solvation:
gbneck2 + pwsasa(surften=0.007)

Codes:
16_koushik_gbsa_3 : cpu pmemd, based on amber16, no cmap
git_3             : cpu pmemd, Git version from Kellon
ambernewSPFP      : gpu pmemd.cuda, problematic

ESURF of 16_koushik_3 and git_3 should match
