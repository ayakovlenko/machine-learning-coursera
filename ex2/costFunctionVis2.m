x = linspace(0, 1, 50);
y1 = 1;
y0 = 0;

costY1 = -y1 * log(x) - (1 - y1) * log(1 - x);
costY0 = -y0 * log(x) - (1 - y0) * log(1 - x);

figure;
hold on;

plot(x, costY1, 'b+');
plot(x, costY0, 'ro');

hold off;

legend('y = 1', 'y = 0')