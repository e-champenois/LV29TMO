import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def findHits(np.ndarray[np.uint16_t, ndim=1, mode='c'] im_u16, np.ndarray[np.float32_t, ndim=1, mode='c'] filt, np.uint16_t thresh, int gaussian_kernel_radius, int median_kernel_radius):
    
    cdef int i, j, k, num_hot_pixels
    cdef int edge_radius = max(gaussian_kernel_radius, median_kernel_radius)
    cdef double d
    cdef np.ndarray[np.uint16_t, ndim=1, mode='c'] X_hot_pixels, Y_hot_pixels
    im_u16, X_hot_pixels, Y_hot_pixels, num_hot_pixels = findHotPixels(im_u16, edge_radius, thresh)
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] im_f32 = gaussFilter(medianFilter(im_u16, X_hot_pixels, Y_hot_pixels, num_hot_pixels, median_kernel_radius), filt, gaussian_kernel_radius, X_hot_pixels, Y_hot_pixels, num_hot_pixels)

    xC = []
    yC = []
    val = []
    
    for k in range(num_hot_pixels):
        i = X_hot_pixels[k]
        j = Y_hot_pixels[k]
        d = im_f32[i*1024+j]
        if d>thresh and \
           d>im_f32[(i-1)*1024+j+1] and d>im_f32[i*1024+j+1] and \
           d>im_f32[(i+1)*1024+j] and d>im_f32[(i+1)*1024+j+1] and \
           d>=im_f32[(i-1)*1024+j-1] and d>=im_f32[(i-1)*1024+j] and \
           d>=im_f32[i*1024+j-1] and d>=im_f32[(i+1)*1024+j-1]:
            xC.append(i)
            yC.append(j)
            val.append(d)
                
    return xC, yC, val

def make_kernel(gaussian_kernel_radius, sigma):
    radius = (np.mgrid[-gaussian_kernel_radius:gaussian_kernel_radius+1, -gaussian_kernel_radius:gaussian_kernel_radius+1]**2).sum(0)
    return np.exp(-radius.flatten()/(2*sigma**2)).astype(np.float32)

@cython.boundscheck(False)
@cython.wraparound(False)
def findHotPixels(np.ndarray[np.uint16_t, ndim=1, mode='c'] im_u16, int edge_radius, np.uint16_t thresh):
    
    cdef int i, j, k=0
    cdef np.ndarray[np.uint16_t, ndim=1, mode='c'] X_hot_pixels=np.empty(1024**2, dtype="uint16"), Y_hot_pixels=np.empty(1024**2, dtype="uint16")
    
    for i in range(edge_radius, 1024-edge_radius):
        for j in range(edge_radius, 1024-edge_radius):
            if im_u16[i*1024+j] > thresh:
                X_hot_pixels[k] = i
                Y_hot_pixels[k] = j
                k += 1
            else:
                im_u16[i*1024+j] = 0
                
    return im_u16, X_hot_pixels[:k], Y_hot_pixels[:k], k

@cython.boundscheck(False)
@cython.wraparound(False)
def gaussFilter(np.ndarray[np.uint16_t, ndim=1, mode='c'] im_u16, np.ndarray[np.float32_t, ndim=1] kern, np.int16_t gaussian_kernel_radius, np.ndarray[np.uint16_t, ndim=1, mode='c'] X_hot_pixels, np.ndarray[np.uint16_t, ndim=1, mode='c'] Y_hot_pixels, np.int64_t num_hot_pixels):
    
    cdef int i, j, x, y
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] out = np.zeros([1048576], dtype="float32")
    cdef float d

    for k in range(num_hot_pixels):
        i = X_hot_pixels[k]
        j = Y_hot_pixels[k]        
        d = 0
        for x in range(-gaussian_kernel_radius, gaussian_kernel_radius+1):
            for y in range(-gaussian_kernel_radius, gaussian_kernel_radius+1):
                d += <float> im_u16[(x+i)*1024 + y + j] * kern[(x+gaussian_kernel_radius)*(2*gaussian_kernel_radius+1) + y + gaussian_kernel_radius]
        out[i*1024+j] = d
    
    return out
    
@cython.boundscheck(False)
@cython.wraparound(False)
def medianFilter(np.ndarray[np.uint16_t, ndim=1, mode='c'] im_u16, np.ndarray[np.uint16_t, ndim=1, mode='c'] X_hot_pixels, np.ndarray[np.uint16_t, ndim=1, mode='c'] Y_hot_pixels, np.int64_t num_hot_pixels, np.int16_t median_kernel_radius):
    
    cdef int idx,i,j,x,y,k
    cdef np.ndarray[np.uint16_t, ndim=1] out = np.zeros(1048576, dtype="uint16")
    cdef np.ndarray[np.uint16_t, ndim=1] a = np.empty(9, dtype="uint16")

    for k in range(num_hot_pixels):
        idx = 0
        i = X_hot_pixels[k]
        j = Y_hot_pixels[k]
        for x in range(-median_kernel_radius, median_kernel_radius+1):
            for y in range(-median_kernel_radius, median_kernel_radius+1):
                a[idx] = im_u16[(i+x)*1024+j+y]
                idx += 1
        out[i*1024+j] = nth_element(a, 2*median_kernel_radius*(median_kernel_radius+1))
    
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def nth_element(np.ndarray[np.uint16_t, ndim=1, mode='c'] a,np.int16_t n):
    
    cdef int l=0,m=8,o=0,p=8,z
    
    while l<m:
        z = a[n]
        while 1:
            while a[o]<z: o+=1
            while z<a[p]: p-=1
            if o<=p:
                a[o],a[p] = a[p],a[o]
                o+=1
                p-=1
            if o>p: break
        if p<n:
            if n<o:
                m=p
            else:
                p=m
            l=o
        else:
            if n<o:
                m=p
            else:
                p=m
            o=l
    
    return a[n]
