clear all;
true_theta = rand()*(2*pi);
true_r = rand()*0.75; % constrain it in r=3/4

pos_true = true_r * [cos(true_theta);sin(true_theta)];

for K=[1 2 3 4 40]
   ref_thetas = linspace(0, 2*pi, K+1);
   x = zeros([2 K]);
   y = zeros([1 K]);
   sigma = 0.3*ones([1 K]);
   for j =1:K
      x(1,j) = cos(ref_thetas(j));
      x(2,j) = sin(ref_thetas(j));
      y(j) = norm(x(:, j)-pos_true);
      noise = mvnrnd(0,sigma(j),1);
      while y(j)+noise < 0 % get valid reading
          noise = mvnrnd(0,sigma(j),1);
      end
      y(j) = y(j) + noise;
   end
   sigma_x = 0.25;
   sigma_y = 0.25;
   
   % plot pieces
   figure;
   plot(x(1, :), x(2,:), 'ob'); hold on;
   plot(pos_true(1), pos_true(2), '+g');
   title(sprintf('plot for k=%i', K));
   xlim([-2 2]);
   ylim([-2 2]);
   % contours functions, takes in a [xe0, ye0] (candidate theta), and finds
   % the LL at that point.
   pdf = @(xe,ye) arrayfun(@(xe0,ye0) map_loglike([xe0;ye0], sigma_x, sigma_y, x,y, sigma),xe,ye);
   g = gca;
   fc = fcontour(pdf,[g.XLim g.YLim]);
   fc.LineStyle='--';
   fc.LevelList = exp(-1:0.1:3);
   colorbar
   hold off;
   disp(map_loglike(pos_true, sigma_x, sigma_y, x,y, sigma));
end


function LL = map_loglike(theta, sig_x,sig_y, x, y, sigma)
    [~, K] = size(x);
    
    p1 = 0;% sum of ln(p(y_i given x_i theta))
    for i=1:K
        p1 = p1 + sigma(i)^-2 * (y(i)-norm(x(:, i)-theta))^2; % data points based
    end
    
    cov = [sig_x^2 0 ; 0 sig_y^2];
    p2 = theta.'*inv(cov)*theta; % ln(p(theta)) part.
    
    LL = p1 + p2;
    LL = 1/(2*K) * LL;
end

