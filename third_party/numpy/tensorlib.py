import struct
import numpy as np


def save_tensor(path, ts, dtype='float32'):
    cell_type = 0
    cell_x = 1
    cell_y = 1
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', cell_type))
        f.write(struct.pack('<i', cell_x))
        f.write(struct.pack('<i', cell_y))
        if 'int32' == dtype:
            type_id = -1
            f.write(struct.pack('<i', type_id))
            while True:
                dim_ts = len(ts.shape)
                if dim_ts > 4:
                    print('Cannot process Tensors beyond 4-D.')
                    return
                elif dim_ts == 4:
                    break
                else:
                    ts = np.expand_dims(ts, axis=0)
            shape_ts = np.shape(ts)
            gros_ts, slis_ts, rows_ts, cols_ts = shape_ts
            f.write(struct.pack('<i', rows_ts))
            f.write(struct.pack('<i', cols_ts))
            f.write(struct.pack('<i', slis_ts))
            f.write(struct.pack('<i', gros_ts))
            ts.astype('int32').tofile(f)
        elif 'float32' == dtype:
            type_id = -2
            f.write(struct.pack('<i', type_id))
            while True:
                dim_ts = len(ts.shape)
                if dim_ts > 4:
                    print('Cannot process Tensors beyond 4-D.')
                    return
                elif dim_ts == 4:
                    break
                else:
                    ts = np.expand_dims(ts, axis=0)
            shape_ts = np.shape(ts)
            gros_ts, slis_ts, rows_ts, cols_ts = shape_ts
            print('shape_ts: ', shape_ts)
            f.write(struct.pack('<i', rows_ts))
            f.write(struct.pack('<i', cols_ts))
            f.write(struct.pack('<i', slis_ts))
            f.write(struct.pack('<i', gros_ts))
            ts.astype('float32').tofile(f)
        else:
            print('Unsupported type.')


def load_tensor(path):
    with open(path, 'rb') as f:
        cell_type = struct.unpack('<i', f.read(4))[0]
        _ = struct.unpack('<i', f.read(4))[0]
        _ = struct.unpack('<i', f.read(4))[0]
        type_id = struct.unpack('<i', f.read(4))[0]
        if cell_type:
            rows_ts = struct.unpack('<i', f.read(4))[0]
            cols_ts = struct.unpack('<i', f.read(4))[0]
            slis_ts = struct.unpack('<i', f.read(4))[0]
            gros_ts = struct.unpack('<i', f.read(4))[0]
            numel_ts = rows_ts * cols_ts * slis_ts * gros_ts
            if type_id == -1:
                ts = np.zeros(numel_ts)
                for n in range(numel_ts):
                    ts[n] = struct.unpack('<i', f.read(4))[0]
            elif type_id == -2:
                ts = np.zeros(numel_ts)
                for n in range(numel_ts):
                    ts[n] = struct.unpack('<f', f.read(4))[0]
            else:
                print('Type error in file loading.')
            return ts.reshape((gros_ts, slis_ts, rows_ts, cols_ts), order='C')
        else:
            rows_ts = struct.unpack('<i', f.read(4))[0]
            cols_ts = struct.unpack('<i', f.read(4))[0]
            slis_ts = struct.unpack('<i', f.read(4))[0]
            gros_ts = struct.unpack('<i', f.read(4))[0]
            if type_id == -1:
                ts = np.fromfile(f, dtype='i').reshape((gros_ts, slis_ts, rows_ts, cols_ts), order='C')
            elif type_id == -2:
                ts = np.fromfile(f, dtype='f').reshape((gros_ts, slis_ts, rows_ts, cols_ts), order='C')
            else:
                print('Type error in file loading.')
            return ts


def save_tensors(path, v, dtype='float32'):
    num_tensors = len(v)
    cell_type = 1
    cell_x = num_tensors
    cell_y = 1
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', cell_type))
        f.write(struct.pack('<i', cell_x))
        f.write(struct.pack('<i', cell_y))
        if 'int32' == dtype:
            type_id = -1
            f.write(struct.pack('<i', type_id))
            for ts in v:
                while True:
                    dim_ts = len(ts.shape)
                    if dim_ts > 4:
                        print('Cannot process Tensors beyond 4-D.')
                        return
                    elif dim_ts == 4:
                        break
                    else:
                        ts = np.expand_dims(ts, axis=0)
                shape_ts = np.shape(ts)
                gros_ts, slis_ts, rows_ts, cols_ts = shape_ts
                f.write(struct.pack('<i', rows_ts))
                f.write(struct.pack('<i', cols_ts))
                f.write(struct.pack('<i', slis_ts))
                f.write(struct.pack('<i', gros_ts))
                ts.astype('int32').tofile(f)
        elif 'float32' == dtype:
            type_id = -2
            f.write(struct.pack('<i', type_id))
            for ts in v:
                while True:
                    dim_ts = len(ts.shape)
                    if dim_ts > 4:
                        print('Cannot process Tensors beyond 4-D.')
                        return
                    elif dim_ts == 4:
                        break
                    else:
                        ts = np.expand_dims(ts, axis=0)
                shape_ts = np.shape(ts)
                gros_ts, slis_ts, rows_ts, cols_ts = shape_ts
                print('shape_ts: ', shape_ts)
                f.write(struct.pack('<i', rows_ts))
                f.write(struct.pack('<i', cols_ts))
                f.write(struct.pack('<i', slis_ts))
                f.write(struct.pack('<i', gros_ts))
                ts.astype('float32').tofile(f)
        else:
            print('Unsupported type.')


def load_tensors(path):
    tensor_group = list()
    with open(path, 'rb') as f:
        cell_type = struct.unpack('<i', f.read(4))[0]
        cell_x = struct.unpack('<i', f.read(4))[0]
        cell_y = struct.unpack('<i', f.read(4))[0]
        type_id = struct.unpack('<i', f.read(4))[0]
        if cell_type:
            num_tensors = cell_x * cell_y
            for cell_iter in range(num_tensors):
                rows_ts = struct.unpack('<i', f.read(4))[0]
                cols_ts = struct.unpack('<i', f.read(4))[0]
                slis_ts = struct.unpack('<i', f.read(4))[0]
                gros_ts = struct.unpack('<i', f.read(4))[0]
                numel_ts = rows_ts * cols_ts * slis_ts * gros_ts
                if type_id == -1:
                    ts = np.zeros(numel_ts)
                    for n in range(numel_ts):
                        ts[n] = struct.unpack('<i', f.read(4))[0]
                elif type_id == -2:
                    ts = np.zeros(numel_ts)
                    for n in range(numel_ts):
                        ts[n] = struct.unpack('<f', f.read(4))[0]
                else:
                    print('Type error in file loading.')
                tensor_group.append(ts.reshape((gros_ts, slis_ts, rows_ts, cols_ts), order='C'))
        else:
            rows_ts = struct.unpack('<i', f.read(4))[0]
            cols_ts = struct.unpack('<i', f.read(4))[0]
            slis_ts = struct.unpack('<i', f.read(4))[0]
            gros_ts = struct.unpack('<i', f.read(4))[0]
            if type_id == -1:
                ts = np.fromfile(f, dtype='i').reshape((gros_ts, slis_ts, rows_ts, cols_ts), order='C')
            elif type_id == -2:
                ts = np.fromfile(f, dtype='f').reshape((gros_ts, slis_ts, rows_ts, cols_ts), order='C')
            else:
                print('Type error in file loading.')
            tensor_group.append(ts)
    return tensor_group


if __name__ == '__main__':
    c = np.arange(120).reshape([5, 4, 2, 3])
    print(c)
    save_tensor('test.bin', c, dtype='int32')
    p = load_tensor('test.bin')
    print(p)
