import numpy as np

# TODO matroid theory används kanske eller annan diskret optimering för att hitta
# TODO integer programming visade sig vara bättre, men om jag vill tillåta 
# vissa chunks att vara större än andra så kan jag använda dynamic programming

def fit_chunk(im_height, im_width, width, height):

    v_slices = int(np.ceil(im_height / height))
    h_slices = int(np.ceil(im_width / width))

    v_diff = height * v_slices - im_height
    h_diff = width * h_slices - im_width

    r_height = int(np.ceil(im_height / v_slices))
    r_width = int(np.ceil(im_width / h_slices))



    print("recommended dims:", "w:", r_width, "h:", r_height)
    print("slices:", "v:", v_slices, "h:", h_slices)
    print("diffs:", v_diff, h_diff)

