function [relres] = evalAlignedUnwarp(x, y, vx, vy)
%evalAlignedUnwarp - Description
%
% Syntax: relres = evalAlignedUnwarp(A, imref)

    g = imgradient(y);
    g = g / max(g(:));
    % align
    [T, ~] = alignLD(g, vx, vy);
    [xx, yy] = meshgrid(1 : size(vx, 2), 1 : size(vy, 1));
    vx = T(1, 1) .* (xx + vx) + T(3, 1) - xx;
    vy = T(2, 2) .* (yy + vy) + T(3, 2) - yy;
    g = imresize(g, size(vx));
    vx = g .* vx;
    vy = g .* vy;
    t = sqrt(vx.^2 + vy.^2);
    relres = mean(t(:));
end