
if [ "$2" = "single"  ];
then
	i=$1
        /Users/yuhua/libffm/ffm-train $3 -p "test_"$i".ffm" "train_"$i".ffm" "model"$i
	
	echo "train logloss"
	/Users/yuhua/libffm/ffm-predict "train_"$i".ffm" "model"$i "output"$i
	echo "test logloss"
	/Users/yuhua/libffm/ffm-predict "test_"$i".ffm" "model"$i "output"$i
	#python logloss.py "output"$i "test_"$i".ffm"	
	exit 0
fi

for (( i=0; i<=$1; i++ ))
do
        /Users/yuhua/libffm/ffm-train $3 -p "test_"$i".ffm" "train_"$i".ffm" "model"$i
	/Users/yuhua/libffm/ffm-predict "test_"$i".ffm" "model"$i "output"$i
	
	#python logloss.py "output"$i "test_"$i".ffm"
done
