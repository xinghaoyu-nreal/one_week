import numpy as np
import matplotlib.pyplot as pl

def plot_repro(ind):
    root_path = "./config/two_image_pose_estimation/repro"
    file_name = root_path + str(ind) + ".txt"
    x, y = np.loadtxt(file_name, delimiter=' ', unpack=True)
    
    values = np.arange(len(x)) / np.double(len(x))
    cmap = pl.cm.jet(values,alpha=0.5)    
    sub = pl.subplot()

    for i in range(x.size):
        color = cmap[i,:]
        pl.plot(x[i], y[i], 'x',  lw=3, mew=3)

    SM = pl.cm.ScalarMappable(pl.cm.colors.Normalize(0.0,x.size), pl.cm.jet)
    SM.set_array(np.arange(x.size));
    cb = pl.colorbar(SM)
    cb.set_label('image index')

    pl.axis('equal')
    pl.grid('on')
    pl.xlabel('error x (pix)')
    pl.ylabel('error y (pix)')

    img_path = root_path + str(ind) + "_error.png"
    pl.savefig(img_path)
    pl.show()

if __name__=="__main__":
    plot_repro(1)
    plot_repro(2)
