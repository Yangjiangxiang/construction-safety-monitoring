from Stream import VideoTracker
 
if __name__ == '__main__':
    # select the video you want to use
    '''
    @video_path<str> -> |视频路径
    @use_frame<list> -> |取值范围[0,1],如果设置为【0，0.5】那么意思就是只使用该视频的前半段。
    @display<bool> -> |是否播放视频。
    '''
    with VideoTracker(video_path='test-mine.mp4',use_frame=[0, 1],save_path='./out.mp4') as vdo_trk:
        vdo_trk.run()