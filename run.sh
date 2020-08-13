#!/bin/bash

# mlagents-learn config/sac/3DBallHard.yaml --run-id=hardball_sac --env=envs/3dballhard_qudrew --num-envs=4 --no-graphics
# mlagents-learn config/sac_transfer/3DBall.yaml --run-id=ball_2f --env=envs/3dball_qudrew --num-envs=4 --no-graphics
# mlagents-learn config/sac_transfer/3DBall.yaml --run-id=ball_1f --env=envs/3dball1f_qudrew --num-envs=4 --no-graphics
mlagents-learn config/sac_transfer/3DBall1fTransfer.yaml --run-id=ball_transfer_2f_noload-model --env=envs/3dball_qudrew --num-envs=4 --no-graphics
# mlagents-learn config/sac_transfer/3DBallHardTransfer.yaml --run-id=transfer_action-enc_linear --env=envs/3dballhard_qudrew --num-envs=4 --no-graphics
# mlagents-learn config/sac_transfer/3DBallHard.yaml --run-id=hardball_action-enc_linear --env=envs/3dballhard_qudrew --num-envs=4 --no-graphics
# mlagents-learn config/sac_transfer/3DBallHardTransfer1.yaml --run-id=sac_transfer_hardball_fixpol_ov --env=envs/3dballhard --num-envs=4 --no-graphics

# mlagents-learn config/sac_transfer/CrawlerStatic.yaml --run-id=oldcs --env=envs/old_crawler_static --num-envs=4 --no-graphics
# mlagents-learn config/sac_transfer/CrawlerStaticTransfer.yaml --run-id=transfer_newcs --env=envs/new_crawler_static --num-envs=4 --no-graphics
# mlagents-learn config/sac_transfer/CrawlerStatic.yaml --run-id=newcs--env=envs/new_crawler_static --num-envs=4 --no-graphics