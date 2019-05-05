clear all
clc
close all;

path='../Data/image2/';
load segments_and_index15.mat;
label=imread(strcat(path,'gt.tif'));
gradient=imread(strcat(path,'gradient.tif'));
gt_t=tabulate(label(:));
cls_numbers=zeros(size(gt_t,1)-1,1);
select_numbers=zeros(size(gt_t,1)-1,1);
cls_sp_numbers=zeros(size(gt_t,1)-1,1); 
real_select_numbers=zeros(size(gt_t,1)-1,3);
small_sp_numbers=zeros(size(gt_t,1)-1,1);
for i=1:size(gt_t,1)-1
    gra_b_stu{i}=zeros(0,4);
    gra_s_stu{i}=zeros(0,4);
end
select_number=2000;
for i=2:size(gt_t,1)
    cls_numbers(i-1)=gt_t(i,2);
    if select_number<=cls_numbers(i-1)*0.4
        select_numbers(i-1)=select_number;
    else
        select_numbers(i-1) = round(cls_numbers(i-1)*0.4);
    end
end
nan_num=0;

%remove nan
for i=1:length(STATS)
    s=STATS(i);
    if s.Area==0
        nan_num=nan_num+1;
    end
end
all_bl=zeros(length(STATS)-nan_num,1);
all_sp_gra=zeros(length(STATS)-nan_num,1);
all_gra_is_small=zeros(length(STATS)-nan_num,1);
no_cen_num=0;
cen_i=1;
% cal all superpixel 
for i=1:length(STATS)
    s=STATS(i);
    if s.Area~=0
        ps=Pixel_index(i).PixelIdxList; %index
        sp_gra=gradient(ps);
        all_sp_gra(cen_i)=sum(sp_gra(:));
        cls_es=label(ps);
        t=tabulate(cls_es(:));
        if size(t,1)==1 && t(1,1)==0 % only unlabeled sample
            continue;
        end
        if t(1,1)==0 % remove unlabeled sample
            t(1,:)=[];
        end
        
        for k=size(t,1):-1:1 % remove the number is 0 (tabulate)
            if t(k,2)==0
                t(k,:)=[];
            end
        end
        
        [m,max_i] = max(t(:,3));
        all_bl(cen_i)=m;
        temp=zeros(size(t,1),4);
        for j=1:size(t,1)
            temp(j,1)=t(j,1); % class label
            temp(j,2)=i;
            temp(j,3)=t(j,2); % the number of this class
            real = round(t(j,2)*select_numbers(t(j,1))/cls_numbers(t(j,1)));
            if real<1
                real=1;
            end
            temp(j,4)=real;
            real_select_numbers(t(j,1),1)=real_select_numbers(t(j,1),1)+real;
        end
        
        for k=size(temp,1):-1:1 % remove the number is 0
            if temp(k,3)==0
                temp(k,:)=[];
            end
        end
        
        for k=1:size(temp,1) % sp number for each class
            cls_sp_numbers(temp(k,1))=cls_sp_numbers(temp(k,1))+1;
        end
        
        stu(cen_i).max_bl=all_bl(cen_i);
        stu(cen_i).sp_index=i;
        stu(cen_i).all_number=temp;
        if all_sp_gra(cen_i)<10 && all_bl(cen_i)==100 % 100% one class
            stu(cen_i).is_gra_small=1;
            small_sp_numbers(temp(1,1))=small_sp_numbers(temp(k,1))+1;
            gra_s_stu{temp(1,1)}=cat(1,gra_s_stu{temp(1,1)},temp);
        else
            stu(cen_i).is_gra_small=0;
            for k=1:size(temp,1)
                gra_b_stu{temp(k,1)} =cat(1,gra_b_stu{temp(k,1)},temp(k,:));
            end
        end
        all_gra_is_small(cen_i)= stu(cen_i).is_gra_small;
        cen_i=cen_i+1;
    end
end

% remove numbers>select_number,determine the select number of each
% superpixel


for i=1:size(gt_t,1)-1
    if real_select_numbers(i,1)>select_numbers(i)
        rm=real_select_numbers(i,1)-select_numbers(i);
        gra_s_rm=min(real_select_numbers(i,1)-select_numbers(i),round(sum(gra_s_stu{i}(:,4))));
        gra_b_rm=rm-gra_s_rm;  
        sum_i=sum(gra_s_stu{i}(:,4));
        if gra_s_rm~=0
            distance_thresh1=15;
            l_rand=randperm(size(gra_s_stu{i},1));
            gra_s=gra_s_stu{i}(l_rand,:);
            while sum(gra_s(:,4))>sum_i-gra_s_rm && size(gra_s,1)>1
                rm_sp=0;
                for j=size(gra_s_stu{i},1)-1:-1:1
                    if sum(gra_s(:,4))<=sum_i-gra_s_rm
                        break;
                    end
                    all_is_near=0;
                    for k=size(gra_s,1):-1:j+1
                        if sp_distance(STATS(gra_s(k,2)).Centroid,STATS(gra_s(j,2)).Centroid)<distance_thresh1
                            all_is_near=all_is_near+1;
                        end
                    end
                    if all_is_near>0
                        gra_s(j,:)=[];
                    end       
                end
                distance_thresh1=distance_thresh1+2;
                l_rand=randperm(size(gra_s,1));
                gra_s=gra_s(l_rand,:);
            end
            gra_s_stu{i}=gra_s;
        end
        
        sum_i2=sum(gra_b_stu{i}(:,4));
        gra_b_rm=min(gra_b_rm,round(0.5*sum_i2));
        if gra_b_rm~=0
            %remove according to numbers
            gra_b=gra_b_stu{i};
            gra_b=sortrows(gra_b,4);
            while sum(gra_b(:,4))>sum_i2-gra_b_rm
                s1=sum(gra_b(:,4));
                gra_b=sortrows(gra_b,4);
                for j=size(gra_b,1)-1:-1:1
                    if sum(gra_b(:,4))<=sum_i2-gra_b_rm
                        break;
                    end
                    if gra_b(j,4)>10 
                        gra_b(j,4)=gra_b(j,4)-round(gra_b(j,4)*0.1);
                    elseif gra_b(j,4)>1 && gra_b(j,4)<11
                         gra_b(j,4)=gra_b(j,4)-1;
                    end
                end
                if s1==sum(gra_b(:,4))
                    break;
                end
            end
            %remove sp numbers according to distance
            distance_thresh2=15;
            while sum(gra_b(:,4))>sum_i2-gra_b_rm && size(gra_b,1)>1
                l2_rand=randperm(size(gra_b,1));
                gra_b=gra_b(l2_rand,:);
                rm_sp2=0;
                for j=size(gra_b_stu{i},1)-1:-1:1
                    if sum(gra_b(:,4))<=sum_i2-gra_b_rm
                        break;
                    end
                    all_is_near=0;
                    for k=size(gra_b,1):-1:j+1
                        if sp_distance(STATS(gra_b(k,2)).Centroid,STATS(gra_b(j,2)).Centroid)<distance_thresh2
                            all_is_near=all_is_near+1;
                        end
                    end
                    if all_is_near>0
                        gra_b(j,:)=[];
                    end       
                end
                distance_thresh2=distance_thresh2+2;
            end
            gra_b_stu{i}=gra_b;
        end
    end
end
% concat all classes
gra_s_stu_new=zeros(0,4);
gra_b_stu_new=zeros(0,4);
for i=1:size(gt_t,1)-1
    gra_s_stu_new=cat(1,gra_s_stu_new,gra_s_stu{i});
    gra_b_stu_new=cat(1,gra_b_stu_new,gra_b_stu{i});
end

% determine the indexs of selected samples
% gradient is big or classes in a pixel is not 100%
select_xyc_all=zeros(sum(gra_s_stu_new(:,4))+sum(gra_b_stu_new(:,4)),3);
all_i=1;
for i=1:size(gra_b_stu_new,1)
    all_i1=all_i;
    xys=Pixel_xy(gra_b_stu_new(i,2)).PixelList;
    indexs=Pixel_index(gra_b_stu_new(i,2)).PixelIdxList;
    for j=size(xys,1):-1:1
        if (label(xys(j,2),xys(j,1))) ~=gra_b_stu_new(i,1)
            xys(j,:)=[];
            indexs(j,:)=[];
        end
    end
    assert(size(xys,1)>0);
    select_number_i=gra_b_stu_new(i,4);
    if select_number_i<2
        r=randperm(size(xys,1),1);
        select_xyc_all(all_i,:)=[xys(r,2),xys(r,1),gra_b_stu_new(i,1)];
        assert(label(xys(r,2),xys(r,1))==gra_b_stu_new(i,1));
        all_i=all_i+1;
        continue;
    end
    % gradient
    gras_n=floor(select_number_i/2);
    ds_n=select_number_i-gras_n;
    gras=gradient(indexs);
    [gras_sort,g_index]=sort(gras);
    step=floor(size(gras,1)/gras_n);
    j1=1;
    select_xy_index=zeros(gras_n,1);
    ii=1;
    for j=1:step:size(indexs,1)-step+1
        while j1<gras_n+1
        indexs1=g_index(j:j+step-1,:);
        if j1==gras_n-1%(j+step*2-1)>size(indexs,1)
            indexs1=g_index(j:end,:);
        end
        r=randperm(size(indexs1,1),1);
        select_xyc_all(all_i,:)=[xys(r+j-1,2),xys(r+j-1,1),gra_b_stu_new(i,1)];
        select_xy_index(ii)=r+j-1;
        ii=ii+1;
        assert(label(xys(r+j-1,2),xys(r+j-1,1))==gra_b_stu_new(i,1));
        all_i=all_i+1;
        j1=j1+1;
        end
    end
    %Avoid selecting duplicate samples
    for ii=1:size(select_xy_index,1)
        xys(ii,:)=[];
        indexs(ii,:)=[];
    end
    all_i2=all_i;
%     all_i2-all_i1
    % distance to centroid
    centroid=STATS(gra_b_stu_new(i,2)).Centroid;
    ds_to_cen=cal_ds(centroid,xys);
    [ds_sort,d_index]=sort(ds_to_cen);
    step2=floor(size(ds_to_cen,1)/ds_n);
    j1=1;
    for j=1:step2:size(indexs,1)-step2+1
        while j1<ds_n+1
        indexs1=d_index(j:j+step2-1,:);
        if j1==ds_n-1%(j+step2*2-1)>size(indexs,1)
            indexs1=d_index(j:end,:);
        end
        r=randperm(size(indexs1,1),1);
        select_xyc_all(all_i,:)=[xys(r+j-1,2),xys(r+j-1,1),gra_b_stu_new(i,1)];
        assert(label(xys(r+j-1,2),xys(r+j-1,1))==gra_b_stu_new(i,1));
        all_i=all_i+1;
        j1=j1+1;
        end
    end 
    all_i3=all_i;
%     all_i3-all_i1
    assert(select_number_i==all_i3-all_i1);
end
% gradients is small and classes in a pixel is not 100%
for i=1:size(gra_s_stu_new,1)
    all_i4=all_i;
    xys=Pixel_xy(gra_s_stu_new(i,2)).PixelList;
    indexs=Pixel_index(gra_s_stu_new(i,2)).PixelIdxList;
    for j=size(xys,1):-1:1
        if (label(xys(j,2),xys(j,1))) ~=gra_s_stu_new(i,1)
            xys(j,:)=[];
            indexs(j,:)=[];
        end
    end
    assert(size(xys,1)>0);
    select_number_i=gra_s_stu_new(i,4);
    if select_number_i<2
        r=randperm(size(xys,1),1);
        select_xyc_all(all_i,:)=[xys(r,2),xys(r,1),gra_s_stu_new(i,1)];
        assert(label(xys(r,2),xys(r,1))==gra_s_stu_new(i,1));
        all_i=all_i+1;
        continue;
    end

    ds_n=select_number_i;
    centroid=STATS(gra_s_stu_new(i,2)).Centroid;
    ds_to_cen=cal_ds(centroid,xys);
    [ds_sort,d_index]=sort(ds_to_cen);
    step=floor(size(ds_to_cen,1)/ds_n);
    j1=1;
    for j=1:step:size(indexs,1)-step+1
        while j1<ds_n+1
        indexs1=d_index(j:j+step-1,:);
        if j1==ds_n-1%(j+step*2-1)>size(indexs,1)
            indexs1=d_index(j:end,:);
        end
        r=randperm(size(indexs1,1),1);
        select_xyc_all(all_i,:)=[xys(r+j-1,2),xys(r+j-1,1),gra_s_stu_new(i,1)];
        assert(label(xys(r+j-1,2),xys(r+j-1,1))==gra_s_stu_new(i,1));
        all_i=all_i+1;
        j1=j1+1;
        end
    end 
    all_i5=all_i;
    assert(select_number_i==all_i5-all_i4);
end
tabulate(select_xyc_all(:,3))
gt_select=zeros(size(label));
for i=1:size(select_xyc_all,1)
    gt_select(select_xyc_all(i,1),select_xyc_all(i,2))=select_xyc_all(i,3);
%     label(select_xyc_all(i,1),select_xyc_all(i,2))
%     select_xyc_all(i,3)
    assert(label(select_xyc_all(i,1),select_xyc_all(i,2))==select_xyc_all(i,3));
    select_xyc_all(i,1)=select_xyc_all(i,1)-1;
    select_xyc_all(i,2)=select_xyc_all(i,2)-1;
end
gt_select=uint8(gt_select);
% result = im1_getRGBimage_colorshow(gt_select);
save sp_xyc_train2000 select_xyc_all;
% imwrite(result,'select_sample.png'); 

function d_to_cen=cal_ds(centroid,xys)
d_to_cen=zeros(size(xys,1),1);
for i=1:size(xys,1)
    d_to_cen(i)=sp_distance(centroid,xys(i,:));
end
end
function  d=sp_distance(a1,a2)
d=sqrt((a1(1)-a2(1))^2+(a1(2)-a2(2))^2);
end
