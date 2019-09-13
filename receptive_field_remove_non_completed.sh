#!/bin/bash
for a in /b/home/uha/hfawaz-datas/temp-dl-tsc/receptive-field/exp/*/; do
	for d in $a*/; do
		for e in $d*/; do
		    for f in $e*/; do
		        for g in $f*/; do
		            for h in $g*/; do
                        for i in $h*/; do
                            for j in $i*/; do
                                for k in $j*/; do
                                    for l in $k*/; do
                                        if [ ! -f "$l/df_metrics.csv" ]; then
                                            echo "removing: $l"
                                            rm  -r $l
                                        fi
                                    done
                                done
                            done
                        done
                    done
                done
            done
		done
	done
done
