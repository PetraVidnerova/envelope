for I in `seq 5 9`
do
    ./r.sh autompg mlp16-8-tanh_$I mlp-tanh & 
done

