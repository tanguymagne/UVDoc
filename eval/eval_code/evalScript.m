function [mean_ms, mean_ld, mean_ad] = evalAD(gtdir, imdir, verbose)
% add LD path
% https://people.csail.mit.edu/celiu/SIFTflow/
% change the path to your SIFTflow folder
addpath(genpath('./SIFTflow'));

res = cell(65, 1);

for k = 1 : 65
    if ~isfile(sprintf('%s/%d.png', gtdir, k))
        fprintf('%s - Not file (ref/input)\n', sprintf('%s/%d.png', gtdir, k))
        t = zeros(2, 5);
        t(1, :) = [k, 1, -1, -1, -1];
        t(2, :) = [k, 2, -1, -1, -1];
        res{k} = t;
        continue 
    end
    rimg = rgb2gray(imread(sprintf('%s/%d.png', gtdir, k)));
    t = zeros(2, 5);
    for m = 1 : 2
        if isfile(sprintf('%s/%d_%d.png', imdir, k, m))
            if verbose
                fprintf('Running %d_%d ... ', k,m)
            end
            ximg = rgb2gray(imread(sprintf('%s/%d_%d.png', imdir, k, m)));
            [rh,rw,~]=size(rimg);
            rimg=imresize(rimg,sqrt(598400/(rh*rw)),'bicubic');
            [rh,rw,~]=size(rimg);
            ximg=imresize(ximg,[rh rw],'bicubic');
            
            [vx, vy] = siftFlow(rimg, ximg);
            ad = evalAlignedUnwarp(ximg, rimg, vx, vy);

            [ms, ld] = evalUnwarp(ximg, rimg, vx, vy);

            t(m, :) = [k, m, ad, ms, ld];
        else
            t(m, :) = [k, m, -1, -1, -1];
            fprintf('%s - Not file\n', sprintf('%s/%d_%d.png', imdir, k, m))
        end

    end
    res{k} = t;
end
res = cell2mat(res);
valres = res(res(:, 3) > 0, :);
avg = mean(valres, 1);

mean_ms = avg(4); 
mean_ld = avg(5);
mean_ad = avg(3);

res = cat(1, res, avg);

save(sprintf('%s/individual_res.txt', imdir), 'res', '-ascii');
end