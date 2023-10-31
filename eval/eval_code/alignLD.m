function [T, relres] = alignLD(g, vx, vy)
    %UNTITLED4 Summary of this function goes here
    %   Detailed explanation goes here
    g = imresize(g, size(vx)) > 0.5;
    [xx1, yy1] = meshgrid(1 : size(g, 2), 1 : size(g, 1));
    xx2 = xx1 + vx;
    yy2 = yy1 + vy;
    % use only high gradient
    xx1 = xx1(g);
    yy1 = yy1(g);
    xx2 = xx2(g);
    yy2 = yy2(g);
    % construct A, B
    t = [xx2, zeros(size(xx2, 1), 1)];
    A1 = reshape(t', 1, [])';
    t = [zeros(size(yy2, 1), 1), yy2];
    A2 = reshape(t', 1, [])';
    A3 = repmat([1; 0], size(xx2, 1), 1);
    A4 = repmat([0; 1], size(xx2, 1), 1);
    A = [A1, A2, A3, A4];
    B = reshape([xx1, yy1]', 1, [])';
    [x, ~, relres] = lsqr(A, B);
    % affine transformation matrix
    T = zeros(3);
    T(1) = x(1);
    T(5) = x(2);
    T(3) = x(3);
    T(6) = x(4);
    T(end) = 1;
    
end
    
    