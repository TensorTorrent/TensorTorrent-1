function [] = savetensor(fileName, data)
% Author: Yuning Jiang
% Date: May. 24 th, 2019
% Description: Save data to a binary file.
% Data type can be double, single, int8, int 16, int32, int 64 or cell.
% CAUTION: Tensor dimension must NOT be greater than 3-D.
% CAUTION: When using a cell type, all units must be of the same type (sizes can vary).

typeid = -1;
typename = 'single';
convert = 0;
celltype = 0;
cellx = 1;
celly = 1;

if isa(data, 'cell')
    [cellx, celly] = size(data);
    A = data{1, 1};
    celltype = 1;
else
    A = data;
end

if isa(A, 'double')  % 'double' 64 (8)
    typeid = -2;
    typename = 'single';
    convert = 1;
elseif isa(A, 'single')  % 'single' 32 (4)
    typeid = -2;
    typename = 'single';
elseif isa(A, 'int32')  % 'int32' 32 (4)
    typeid = -1;
    typename = 'int32';
end

fileID = fopen(fileName, 'w');
fwrite(fileID, celltype, 'int32');
fwrite(fileID, cellx, 'int32');
fwrite(fileID, celly, 'int32');
fwrite(fileID, typeid, 'int32');

if celltype
    for x = 1 : cellx
        for y = 1 : celly
            [rows, cols, slis, gros] = size(data{x, y});
            fwrite(fileID, rows, 'int32');
            fwrite(fileID, cols, 'int32');
            fwrite(fileID, slis, 'int32');
            fwrite(fileID, gros, 'int32');
            if convert
                fwrite(fileID, single(permute(data{x, y}, [2, 1, 3, 4])), typename);
            else
                fwrite(fileID, permute(data{x, y}, [2, 1, 3, 4]), typename);
            end
        end
    end
else
    [rows, cols, slis, gros] = size(data);
    fwrite(fileID, rows, 'int32');
    fwrite(fileID, cols, 'int32');
    fwrite(fileID, slis, 'int32');
    fwrite(fileID, gros, 'int32');
    if convert
        fwrite(fileID, single(permute(data, [2, 1, 3, 4])), typename);
    else
        fwrite(fileID, permute(data, [2, 1, 3, 4]), typename);
    end
end

fclose(fileID);

end
