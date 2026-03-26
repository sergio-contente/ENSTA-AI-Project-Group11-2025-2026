#!/bin/bash
# Download HM3D validation dataset for CoIN evaluation

# Create directory
mkdir -p home/ensta/data/hm3d/val

# Download ALL 4 required files
wget https://api.matterport.com/resources/habitat/hm3d-val-glb-v0.2.tar           # 4G - 3D meshes
wget https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar       # 3.3G - Nav meshes
wget https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar    # 2G - Semantic annotations
wget https://api.matterport.com/resources/habitat/hm3d-val-semantic-configs-v0.2.tar   # 40K - Semantic configs

# Extract to correct location
tar -xf hm3d-val-glb-v0.2.tar -C home/ensta/data/hm3d/
tar -xf hm3d-val-habitat-v0.2.tar -C home/ensta/data/hm3d/
tar -xf hm3d-val-semantic-annots-v0.2.tar -C home/ensta/data/hm3d/
tar -xf hm3d-val-semantic-configs-v0.2.tar -C home/ensta/data/hm3d/

# Remove tar files after extraction to save space
rm -f hm3d-val-*.tar

echo "HM3D validation dataset downloaded and extracted successfully!"