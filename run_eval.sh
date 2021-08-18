#!/bin/bash

echo -n "" > eval_results.txt

# Eval

mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc2-str05-8-9 --env=envs/mac_suite/basic-disc2
mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc4-str05-8-9 --env=envs/mac_suite/basic-disc4
mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc8-str05-8-9 --env=envs/mac_suite/basic-disc8
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn2-str05-8-9 --env=envs/mac_suite/basic-disc2
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn4-str05-8-9 --env=envs/mac_suite/basic-disc4
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn8-str05-8-9 --env=envs/mac_suite/basic-disc8

mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc2-str05-8-9 --env=envs/mac_suite/laser-disc2
mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc4-str05-8-9 --env=envs/mac_suite/laser-disc4
mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc8-str05-8-9 --env=envs/mac_suite/laser-disc8
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn2-str05-8-9 --env=envs/mac_suite/laser-disc2
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn4-str05-8-9 --env=envs/mac_suite/laser-disc4
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn8-str05-8-9 --env=envs/mac_suite/laser-disc8

mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc2-str10-8-9 --env=envs/mac_suite/worm-disc2
mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc4-str10-8-9 --env=envs/mac_suite/worm-disc4
mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc8-str10-8-9 --env=envs/mac_suite/worm-disc8
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn2-str10-8-9 --env=envs/mac_suite/worm-disc2
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn4-str10-8-9 --env=envs/mac_suite/worm-disc4
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn8-str10-8-9 --env=envs/mac_suite/worm-disc8

mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc2-str10-8-9 --env=envs/mac_suite/crawl-disc2
mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc4-str10-8-9 --env=envs/mac_suite/crawl-disc4
mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc8-str10-8-9 --env=envs/mac_suite/crawl-disc8
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn2-str10-8-9 --env=envs/mac_suite/crawl-disc2
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn4-str10-8-9 --env=envs/mac_suite/crawl-disc4
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn8-str10-8-9 --env=envs/mac_suite/crawl-disc8


# A

mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc2-str05-8-11a --env=envs/mac_suite/basic-disc2
mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc4-str05-8-11a --env=envs/mac_suite/basic-disc4
mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc8-str05-8-11a --env=envs/mac_suite/basic-disc8
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn2-str05-8-11a --env=envs/mac_suite/basic-disc2
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn4-str05-8-11a --env=envs/mac_suite/basic-disc4
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn8-str05-8-11a --env=envs/mac_suite/basic-disc8

mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc2-str05-8-11a --env=envs/mac_suite/laser-disc2
mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc4-str05-8-11a --env=envs/mac_suite/laser-disc4
mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc8-str05-8-11a --env=envs/mac_suite/laser-disc8
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn2-str05-8-11a --env=envs/mac_suite/laser-disc2
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn4-str05-8-11a --env=envs/mac_suite/laser-disc4
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn8-str05-8-11a --env=envs/mac_suite/laser-disc8

mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc2-str10-8-11a --env=envs/mac_suite/worm-disc2
mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc4-str10-8-11a --env=envs/mac_suite/worm-disc4
mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc8-str10-8-11a --env=envs/mac_suite/worm-disc8
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn2-str10-8-11a --env=envs/mac_suite/worm-disc2
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn4-str10-8-11a --env=envs/mac_suite/worm-disc4
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn8-str10-8-11a --env=envs/mac_suite/worm-disc8

mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc2-str10-8-11a --env=envs/mac_suite/crawl-disc2
mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc4-str10-8-11a --env=envs/mac_suite/crawl-disc4
mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc8-str10-8-11a --env=envs/mac_suite/crawl-disc8
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn2-str10-8-11a --env=envs/mac_suite/crawl-disc2
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn4-str10-8-11a --env=envs/mac_suite/crawl-disc4
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn8-str10-8-11a --env=envs/mac_suite/crawl-disc8


# B

mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc2-str05-8-11b --env=envs/mac_suite/basic-disc2
mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc4-str05-8-11b --env=envs/mac_suite/basic-disc4
mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc8-str05-8-11b --env=envs/mac_suite/basic-disc8
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn2-str05-8-11b --env=envs/mac_suite/basic-disc2
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn4-str05-8-11b --env=envs/mac_suite/basic-disc4
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn8-str05-8-11b --env=envs/mac_suite/basic-disc8

mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc2-str05-8-11b --env=envs/mac_suite/laser-disc2
mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc4-str05-8-11b --env=envs/mac_suite/laser-disc4
mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc8-str05-8-12b --env=envs/mac_suite/laser-disc8
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn2-str05-8-11b --env=envs/mac_suite/laser-disc2
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn4-str05-8-11b --env=envs/mac_suite/laser-disc4
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn8-str05-8-11b --env=envs/mac_suite/laser-disc8

mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc2-str10-8-12b --env=envs/mac_suite/worm-disc2
mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc4-str10-8-12b --env=envs/mac_suite/worm-disc4
mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc8-str10-8-11b --env=envs/mac_suite/worm-disc8
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn2-str10-8-11b --env=envs/mac_suite/worm-disc2
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn4-str10-8-11b --env=envs/mac_suite/worm-disc4
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn8-str10-8-11b --env=envs/mac_suite/worm-disc8

mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc2-str10-8-12b --env=envs/mac_suite/crawl-disc2
mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc4-str10-8-11b --env=envs/mac_suite/crawl-disc4
mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc8-str10-8-11b --env=envs/mac_suite/crawl-disc8
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn2-str10-8-11b --env=envs/mac_suite/crawl-disc2
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn4-str10-8-12b --env=envs/mac_suite/crawl-disc4
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn8-str10-8-12b --env=envs/mac_suite/crawl-disc8


# C

mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc2-str05-8-11c --env=envs/mac_suite/basic-disc2
mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc4-str05-8-11c --env=envs/mac_suite/basic-disc4
mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc8-str05-8-11c --env=envs/mac_suite/basic-disc8
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn2-str05-8-11c --env=envs/mac_suite/basic-disc2
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn4-str05-8-11c --env=envs/mac_suite/basic-disc4
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn8-str05-8-11c --env=envs/mac_suite/basic-disc8

mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc2-str05-8-11c --env=envs/mac_suite/laser-disc2
mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc4-str05-8-11c --env=envs/mac_suite/laser-disc4
mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc8-str05-8-11c --env=envs/mac_suite/laser-disc8
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn2-str05-8-11c --env=envs/mac_suite/laser-disc2
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn4-str05-8-12c --env=envs/mac_suite/laser-disc4
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn8-str05-8-11c --env=envs/mac_suite/laser-disc8

mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc2-str10-8-11c --env=envs/mac_suite/worm-disc2
mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc4-str10-8-11c --env=envs/mac_suite/worm-disc4
mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc8-str10-8-11c --env=envs/mac_suite/worm-disc8
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn2-str10-8-12c --env=envs/mac_suite/worm-disc2
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn4-str10-8-12c --env=envs/mac_suite/worm-disc4
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn8-str10-8-11c --env=envs/mac_suite/worm-disc8

mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc2-str10-8-11c --env=envs/mac_suite/crawl-disc2
mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc4-str10-8-11c --env=envs/mac_suite/crawl-disc4
mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc8-str10-8-11c --env=envs/mac_suite/crawl-disc8
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn2-str10-8-11c --env=envs/mac_suite/crawl-disc2
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn4-str10-8-11c --env=envs/mac_suite/crawl-disc4
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn8-str10-8-11c --env=envs/mac_suite/crawl-disc8


# D

mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc2-str05-8-11d --env=envs/mac_suite/basic-disc2
mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc4-str05-8-11d --env=envs/mac_suite/basic-disc4
mlagents-learn config/sac/BasicDiverse.yaml --force --run-id=basic-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-disc8-str05-8-11d --env=envs/mac_suite/basic-disc8
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn2-str05-8-11d --env=envs/mac_suite/basic-disc2
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn4-str05-8-11d --env=envs/mac_suite/basic-disc4
mlagents-learn config/sac/BasicDiverse_diayn.yaml --force --run-id=basic-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-basic-diayn8-str05-8-11d --env=envs/mac_suite/basic-disc8

mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc2-str05-8-11d --env=envs/mac_suite/laser-disc2
mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc4-str05-8-11d --env=envs/mac_suite/laser-disc4
mlagents-learn config/sac/LaserMazeDiverse.yaml --force --run-id=laser-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-disc8-str05-8-11d --env=envs/mac_suite/laser-disc8
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn2-str05-8-11d --env=envs/mac_suite/laser-disc2
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn4-str05-8-11d --env=envs/mac_suite/laser-disc4
mlagents-learn config/sac/LaserMazeDiverse_diayn.yaml --force --run-id=laser-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-laser-diayn8-str05-8-11d --env=envs/mac_suite/laser-disc8

mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc2-str10-8-11d --env=envs/mac_suite/worm-disc2
mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc4-str10-8-11d --env=envs/mac_suite/worm-disc4
mlagents-learn config/sac/WormDiverse.yaml --force --run-id=worm-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-disc8-str10-8-11d --env=envs/mac_suite/worm-disc8
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn2-str10-8-11d --env=envs/mac_suite/worm-disc2
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn4-str10-8-11d --env=envs/mac_suite/worm-disc4
mlagents-learn config/sac/WormDiverse_diayn.yaml --force --run-id=worm-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-worm-diayn8-str10-8-11d --env=envs/mac_suite/worm-disc8

mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc2-str10-8-11d --env=envs/mac_suite/crawl-disc2
mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc4-str10-8-11d --env=envs/mac_suite/crawl-disc4
mlagents-learn config/sac/CrawlerDiverse.yaml --force --run-id=crawl-disc8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-disc8-str10-8-11d --env=envs/mac_suite/crawl-disc8
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn2-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn2-str10-8-11d --env=envs/mac_suite/crawl-disc2
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn4-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn4-str10-8-11d --env=envs/mac_suite/crawl-disc4
mlagents-learn config/sac/CrawlerDiverse_diayn.yaml --force --run-id=crawl-diayn8-inf --initialize-from=/Users/kolby.nottingham/ml-agents-cloud-internal/results/kolby-crawl-diayn8-str10-8-11d --env=envs/mac_suite/crawl-disc8
