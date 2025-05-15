__kernel void relu_kernel(__global int* data, int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        data[gid] = max(data[gid], 0);
    }
}