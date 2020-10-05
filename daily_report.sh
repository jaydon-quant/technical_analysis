#!/usr/bin/env bash
source $HOME/.bash_profile
source $conda activate quant3
python $HOME/dev/technical_analysis/run.py daily_report
