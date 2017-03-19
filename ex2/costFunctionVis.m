x = linspace(0, 1, 50);
y1 = -log(x);
y0 = -log(1 - x);

figure;
hold on;

plot(x, y1, 'b+');
plot(x, y0, 'ro');

hold off;

legend('y = 1, -log(x)', 'y = 0, -log(1 - x)')
