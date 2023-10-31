function [mean_ms, mean_ad] = evalScriptUVDoc(gtdir, imdir, verbose)
% add LD path
% https://people.csail.mit.edu/celiu/SIFTflow/
% change the path to your SIFTflow folder
addpath(genpath('./SIFTflow'));

res = cell(50, 1);

for k = 0 : 49
    if ~isfile(sprintf('%s/%05d.png', gtdir, k))
        fprintf('%s - Not file (ref/input)\n', sprintf('%s/%05d.png', gtdir, k))
        res{k + 1} =[k, -1, -1, -1];
        continue 
    end
    rimg = rgb2gray(imread(sprintf('%s/%05d.png', gtdir, k)));
    t = zeros(4);
    if isfile(sprintf('%s/%05d.png', imdir, k))
        if verbose
            fprintf('Running %05d ... ', k)
        end
        ximg = rgb2gray(imread(sprintf('%s/%05d.png', imdir, k)));
        [rh,rw,~]=size(rimg);
        rimg=imresize(rimg,sqrt(598400/(rh*rw)),'bicubic');
        [rh,rw,~]=size(rimg);
        ximg=imresize(ximg,[rh rw],'bicubic');
        
        [vx, vy] = siftFlow(rimg, ximg);
        ad = evalAlignedUnwarp(ximg, rimg, vx, vy);

        [ms, ld] = evalUnwarp(ximg, rimg, vx, vy);

        t = [k, ad, ms, ld];
    else
        t = [k, -1, -1, -1];
        fprintf('%s - Not file\n', sprintf('%s/%05d.png', imdir, k))
    end

    res{k + 1} = t;
end
res = cell2mat(res);
valres = res(res(:, 3) > 0, :);
avg = mean(valres, 1);

mean_ms = avg(3); 
mean_ld = avg(4);
mean_ad = avg(2);

res = cat(1, res, avg);

save(sprintf('%s/individual_res.txt', imdir), 'res', '-ascii');
end