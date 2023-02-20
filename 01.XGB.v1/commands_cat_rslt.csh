
rm ./reference/all_results.csv

foreach f (64 120 137 261 507 582 779 782 821 867)
 echo "-----seed: $f -----" # >> out
#  python exec_xgb_best.py $f | tail -n 1 >> out2
 cat ./reference/results.$f.csv >> ./reference/all_results.csv
end
