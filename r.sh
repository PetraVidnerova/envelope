LABEL=$2
DATA=$1
MODEL=$3 

echo "Data: $DATA"
echo "LABEL: $LABEL"
echo "Model: $MODEL"

python test.py  quantiles --label ${LABEL} ${DATA} 0.1 0.9 ${MODEL} > /dev/null &
python test.py  quantiles --label ${LABEL} ${DATA} 0.2 0.8 ${MODEL} > /dev/null &
python test.py  quantiles --label ${LABEL} ${DATA} 0.3 0.7 ${MODEL} > /dev/null &

wait

python test.py simple-model --label ${LABEL} ${DATA} ${MODEL} > ${DATA}_${LABEL}_plain.txt &
python test.py trimmed-model --label ${LABEL} ${DATA} ${MODEL}_${DATA}_${LABEL}_y_0.1.npy ${MODEL}_${DATA}_${LABEL}_y_0.9.npy ${MODEL} > ${DATA}_${LABEL}_trimmed_0.1_0.9.txt &
python test.py trimmed-model --label ${LABEL} ${DATA} ${MODEL}_${DATA}_${LABEL}_y_0.2.npy ${MODEL}_${DATA}_${LABEL}_y_0.8.npy ${MODEL} > ${DATA}_${LABEL}_trimmed_0.2_0.8.txt &
python test.py trimmed-model --label ${LABEL} ${DATA} ${MODEL}_${DATA}_${LABEL}_y_0.3.npy ${MODEL}_${DATA}_${LABEL}_y_0.7.npy ${MODEL} > ${DATA}_${LABEL}_trimmed_0.3_0.7.txt &

