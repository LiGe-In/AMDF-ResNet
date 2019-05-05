% VL_DEMO_SLICT  Demo: SLIC superpixels
clear all;
clc;

path='../Data/image2/';
filename='rgb.png';
savename=filename;
savename(end-3:end)=[];
spname=strcat(savename,'_sp15','.tif');

im_part = imread(strcat(path,filename));
im_part = im2single(im_part) ;
image(im_part) ;
axis equal off tight ;

regionSize=15;
regularizer = 0.1 ;
segments = vl_slic(im_part, regionSize, regularizer, 'verbose') ;
L = bwlabel(segments);
segments=segments+1;
STATS=regionprops(segments,'basic');
Pixel_xy=regionprops(segments,'PixelList');
Pixel_index=regionprops(segments,'PixelIdxList');
[sx,sy]=vl_grad(double(segments), 'type', 'forward') ;
s = find(sx | sy) ;
imp = im_part ;
imp([s s+numel(im_part(:,:,1)) s+2*numel(im_part(:,:,1))]) = 255 ;
sp=double(imp);
imwrite(sp,strcat(path,spname));

save segments_and_index15 segments STATS Pixel_xy Pixel_index;

