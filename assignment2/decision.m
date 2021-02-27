%Nisarg Jain-17ucc039
clear all;
clc;
x = readtable('diabetes.csv','Format','auto');
X = x{:,:};
T = X(:, 1:8);
Y = X(:,9);
%fitting tree
tree = fitctree(T, Y);
%calculating loss
l = loss(tree, T, Y);
fprintf("loss without pruning: %f\n", l);
%checking loss if tree is pruned, POST pruning method
lp = loss(tree, T, Y, 'Subtrees', 1);
fprintf("loss with level 1 pruning: %f\n", lp);
%since loss is increasing we won't prune
%view(tree1);
fprintf("since loss is increasing we won't prune normal model \ncreating a crossvalidated model \n");
%creating a crossvalidated model 
tree1 = fitctree(T,Y, 'CrossVal','on');
ls = kfoldLoss(tree1,'mode','average');
fprintf("loss without pruning: %f\n", ls);
%checking for pruning
checkloss = kfoldLoss(tree1,'mode','individual');
[~, index] = min(checkloss);
tree2 = tree1.Trained{index};
[~,~,~,bestlevel] = loss(tree2, T, Y, 'SubTrees','All','TreeSize','min');
fprintf("bestlevel for pruning: %f\n", bestlevel);
lsp = loss(tree2, T, Y, 'SubTrees',bestlevel);
fprintf("loss with bestlevel pruning: %f\n", lsp);