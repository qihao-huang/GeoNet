import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

if __name__=="__main__":
    print(os.getcwd())

    file_name = "/userhome/34/h3567721/projects/Depth/GeoNet/log/log_depth_geo_xyz.txt"
    with open(file_name) as file_in:
        content = file_in.readlines()
        loss = []
        for line in content:
            # Iteration: [ 114900] | Time: 0.3317s/iter | Loss: 2.112
            cur_line = line.strip()
            # print(cur_line)
            if cur_line.startswith("Iteration"):
                try:
                    loss.append(float(cur_line.split(" ")[-1]))
                except Exception as e:
                    print(e) 

    plt.figure(figsize = (25,5))
    plt.plot(loss)
    plt.xlabel('Iteration * 100')
    plt.ylabel('Loss')
    plt.savefig("geo_xyz_loss_curve.jpg")