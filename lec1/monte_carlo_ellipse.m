% 使用蒙特卡洛方法计算椭圆面积并绘制图形
% 椭圆方程: (x/a)^2 + (y/b)^2 <= 1

a = 3;  % 半长轴
b = 2;  % 半短轴
N = 100000;  % 随机点数

x_rand = (2*a)*rand(N,1) - a;
y_rand = (2*b)*rand(N,1) - b;

inside = (x_rand.^2 / a^2 + y_rand.^2 / b^2) <= 1; %判断是否在椭圆内部

area_exact = pi * a * b;
area_mc = (sum(inside) / N) * (4 * a * b);

fprintf('精确面积: %.4f\n', area_exact);
fprintf('蒙特卡洛面积: %.4f\n', area_mc);
fprintf('误差: %.4f\n', abs(area_mc - area_exact));

%
figure;
hold on;

% 绘制椭圆
theta = linspace(0, 2*pi, 100);
x_ellipse = a * cos(theta);
y_ellipse = b * sin(theta);
plot(x_ellipse, y_ellipse, 'b-', 'LineWidth', 2);

% 绘制随机点
scatter(x_rand(inside), y_rand(inside), 10, 'r.', 'MarkerEdgeAlpha', 0.5);
scatter(x_rand(~inside), y_rand(~inside), 10, 'b.', 'MarkerEdgeAlpha', 0.5);

axis equal;
xlim([-a-0.5, a+0.5]);
ylim([-b-0.5, b+0.5]);
xlabel('x');
ylabel('y');
title('蒙特卡洛方法计算椭圆面积');
legend('椭圆边界', '内部点', '外部点');
grid on;
hold off;