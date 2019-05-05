% function [R,mag,angle]=scale(Image)
clear 
clc
path='../Data/image2/'
Image=imread(strcat(path,'rgb.png'));
Image=im2double(Image) ;
if (size(Image,3) > 1)
  Image = rgb2gray( Image ) ;
end
Image=Image-min(Image(:)) ;
Image=Image/max(Image(:)) ;
% figure,imshow(Image);

s=1;
sigma=2*(2^(1/3))^(s-1); %2
 r=round(2*sigma);%4
[x,y]=meshgrid(-r:r,-r:r); 
W=exp(-(abs(x) +abs(y))/(sigma));
 W(W<eps*max(W(:))) = 0;
 if sum(W(:))~=0;
        W=W/sum(W(:));
 end  
    W11=zeros(2*r+1,2*r+1);
    W21=zeros(2*r+1,2*r+1);
    W13=zeros(2*r+1,2*r+1);
    W23=zeros(2*r+1,2*r+1);
    
    W11(r+2:2*r+1,:)=W(r+2:2*r+1,:);
    W21(1:r,:)=W(1:r,:);
    W13(:,r+2:2*r+1)=W(:,r+2:2*r+1);
    W23(:,1:r)=W(:,1:r);
    
    M11=imfilter(Image,W11,'replicate');
    M21=imfilter(Image,W21,'replicate');
    M13=imfilter(Image,W13,'replicate');
    M23=imfilter(Image,W23,'replicate');         
 Gx=log(abs(M13./M23));            
 Gy=log(abs(M11./M21));             
 mag(:,:,s)=sqrt(Gx.^2+Gy.^2);
 angle(:,:,s)=mod(atan2(Gy,Gx)+2*pi,2*pi);
gus=round(2*sqrt(2)*sigma);
h= fspecial('gaussian',[2*gus+1 2*gus+1],sqrt(2)*sigma);
A=imfilter((Gx).^2,h,'replicate'); 
B=imfilter((Gy).^2,h,'replicate'); 
C=imfilter((Gx).*(Gy),h,'replicate');
aa=(sigma^2)^2*(A.*B-C.*C)-0.04*(A+B).^2; 
imwrite(aa,strcat(path,'gradient.tif'));
% figure,imshow(aa);
