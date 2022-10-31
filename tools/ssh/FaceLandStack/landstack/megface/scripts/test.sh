export MGB_DISABLE_SET_VALUE_WARN=1

#landmark
#mdl errormodel.py --modelpath /unsullied/sharefs/xiongpengfei/Isilon-alignmentModel/3rdparty/Model24/FaceModel/model/src/lmk.large.v1.170501/model.pkl --oprnames pred,prob --ldname megface_xlarge --iscolor True


# detect
#videodir=/unsullied/sharefs/xiongpengfei/Isilon-alignmentModel/3dlandmark/avatar/datasets/video2
#mdl detectvideo.py -v $videodir/vid.avi -do $videodir/test.det -vo $videodir/test.mp4 -f 0 -g gpu0

#mdl trackvideo.py -v $videodir/test.mp4 -t $videodir/test.det -ot $videodir/test.track -m /unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/temporary/model.large/lmk_large170501 -f pred,prob -oi $videodir/test_frame -ov $videodir/test_track.mp4 -oj $videodir/test_track


# track 
#mdl trackvideo.py -v ../videos/track/track2.mp4 -t ../videos/track/track2.det -ot track2_base.det -ov track2_base.mp4 -m /unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/temporary/model.fast/170707-track-4.9ms/epoch213.bin -f pred,prob -c False
#mdl trackvideo.py -v ../videos/track/trans2.mp4 -t ../videos/track/trans2.det -ot trans2_base.det -ov trans2_base.mp4 -m /unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/temporary/model.fast/170707-track-4.9ms/epoch213.bin -f pred,prob -c False

#mdl trackvideo.py -v ../videos/track/track2.mp4 -t ../videos/track/track2.det -ot track2_10.det -ov track2_10.mp4 -m /unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/release/lmk.track.v4.fast.170930/best-714-prob2 -f s-pred,s-prob -c False
#mdl trackvideo.py -v ../videos/track/trans2.mp4 -t ../videos/track/trans2.det -ot trans2_10.det -ov trans2_10.mp4 -m /unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/release/lmk.track.v4.fast.170930/best-714-prob2 -f s-pred,s-prob -c False

#mdl trackvideo.py -v ../videos/track/track2.mp4 -t ../videos/track/track2.det -ot track2_15.det -ov track2_15.mp4 -m /unsullied/sharefs/xiongpengfei/Isilon-alignmentModel/landmark/proj/postfilter/config/xiongpengfei.0831neg1track15.incepmut.halfallpadmore.stride2.freeze1e3.postconv1/train_log/models/best-621 -f s-pred,s-prob -c False
#mdl trackvideo.py -v ../videos/track/trans2.mp4 -t ../videos/track/trans2.det -ot trans2_15.det -ov trans2_15.mp4 -m /unsullied/sharefs/xiongpengfei/Isilon-alignmentModel/landmark/proj/postfilter/config/xiongpengfei.0831neg1track15.incepmut.halfallpadmore.stride2.freeze1e3.postconv1/train_log/models/best-621 -f s-pred,s-prob -c False

#mdl trackvideo.py -v ../videos/track/track2.mp4 -t ../videos/track/track2.det -ot track2_20.det -ov track2_20.mp4 -m /unsullied/sharefs/xiongpengfei/Isilon-alignmentModel/landmark/proj/postfilter/config/xiongpengfei.0831neg1track.incepmut.halfallpadmore.stride2.freeze1e3.postconv1/train_log/models/best-493 -f s-pred,s-prob -c False
#mdl trackvideo.py -v ../videos/track/trans2.mp4 -t ../videos/track/trans2.det -ot trans2_20.det -ov trans2_20.mp4 -m /unsullied/sharefs/xiongpengfei/Isilon-alignmentModel/landmark/proj/postfilter/config/xiongpengfei.0831neg1track.incepmut.halfallpadmore.stride2.freeze1e3.postconv1/train_log/models/best-493 -f s-pred,s-prob -c False

mdl mgftrackvideo.py -v /unsullied/sharefs/_research_facelm/Isilon-datashare/testvideos/track/trans2.mp4 -ot 1.det -ov 1.mp4 -cf tracker.mobile.v4.pose.171031.conf 



