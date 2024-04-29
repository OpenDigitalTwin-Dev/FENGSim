function mat2coo()

A = load('/Users/zhengli/data/Mat_4K_001.mat');

[row, col, val] = find(cell2mat(struct2cell(A)));

fid = fopen('4K.dat', 'a+');
fprintf(fid, '%d %d %d \r\n', uint32(max(row)),uint32(max(col)),uint32(length(val)));

for i=1:length(val)
    fprintf(fid, '%d %d %g\r\n', uint32(row(i)), uint32(col(i)), val(i));
end
fclose(fid);



