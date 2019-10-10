function [data] = loadtensor(fileName)

fileID = fopen(fileName);
celltype = fread(fileID, 1, 'int32');
cellx = fread(fileID, 1, 'int32');
celly = fread(fileID, 1, 'int32');
typeid = fread(fileID, 1, 'int32');

if typeid == -1  % 'int32' 32 (4)
    typename = 'int32';
elseif typeid == -2  % 'single' 32 (4)
    typename = 'single';
else
    typename = 'single';
    fprintf('\nType error in file loading.\n');
end

if celltype
    data = cell(cellx, celly);
    for x = 1 : cellx
        for y = 1 : celly
            rows = fread(fileID, 1, 'int32');
            cols = fread(fileID, 1, 'int32');
            slis = fread(fileID, 1, 'int32');
            gros = fread(fileID, 1, 'int32');
            temp = fread(fileID, rows * cols * slis * gros, typename);
            data{x, y} = permute(reshape(temp, [cols, rows, slis, gros]), [2, 1, 3, 4]);
        end
    end
else
    rows = fread(fileID, 1, 'int32');
    cols = fread(fileID, 1, 'int32');
    slis = fread(fileID, 1, 'int32');
    gros = fread(fileID, 1, 'int32');
    temp = fread(fileID, rows * cols * slis * gros, typename);
    data = permute(reshape(temp, [cols, rows, slis, gros]), [2, 1, 3, 4]);
end

fclose(fileID);

end
