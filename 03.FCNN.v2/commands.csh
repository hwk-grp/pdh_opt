
foreach f (64 120 137 261 507 582 779 782 821 867)
 echo "-----seed: $f -----" >> out
 python exec_fnn.py $f | tail -n 1 >> out
 mv model.0.pt model.$f.pt
 mv results.csv results.$f.csv
end
