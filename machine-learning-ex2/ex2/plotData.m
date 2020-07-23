function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
pos = find(y==1);     %Find indices where probability is 1,returns a vector  (#positive)
neg = find(y==0);     %Find indices where probability is 0,returns a vector  (#negative)

%Now to plot and as X is Mx2 matrix so-
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);   % X(pos,1) are all elements of the first column of X at rows where the condition y==1 is met.Similarly for neg as well.
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','y','MarkerSize',7);






% =========================================================================



hold off;

end
