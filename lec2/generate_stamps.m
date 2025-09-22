function data = generate_stamps(N)
    mu0 = -1;
    mu1 = 1;
    sigma = 0.5;

    % x|y=0 ~ N(mu0, sigma^2)
    N0 = fix(N/2);
    x0 = randn(N0,1) * sigma + mu0;
    y0 = zeros(N0,1);

    % x|y=1 ~ N(mu1, sigma^2)
    x1 = randn(N-N0,1) * sigma + mu1;
    y1 = ones(N-N0,1);
    
    % Combine the two halves
    data = [x0, y0; x1, y1];
end