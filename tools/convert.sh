for i in `ls *.MOV`
do
    ffmpeg -i ${i} -vcodec libx264 -preset fast -crf 22 -y -acodec copy `echo ${i} | sed 's/MOV/mp4/g'`
done

