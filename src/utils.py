#import necessary modules
import numpy as np
import sys
from numpy import unravel_index
from netCDF4 import Dataset
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import io
from urllib.request import urlopen, Request
from PIL import Image
import tensorflow as tf
import pandas as pd

getDate = lambda x: x.split(" ")[0]

def mapGrid(FILE_NAME,d_lat=0.025,d_lon=0.025,llcrn=(-10.56, 51.39),urcrn=(-5.34, 55.43)):
  
  FILE_NAME=FILE_NAME.strip()
  if not os.path.isfile(FILE_NAME):
    print("The file does not exist!")
    return
    
  if 'NO2' in FILE_NAME:
      #this is how you access the data tree in an hdf5 file
      SDS_NAME='nitrogendioxide_tropospheric_column'    
  elif 'AER_AI' in FILE_NAME:
      SDS_NAME='aerosol_index_354_388'
  ds=Dataset(FILE_NAME,'r')
  grp='PRODUCT'        
  lat= ds.groups[grp].variables['latitude'][0][:][:]
  lon= ds.groups[grp].variables['longitude'][0][:][:]
  data= ds.groups[grp].variables[SDS_NAME]      
  alt = ds.groups[grp].groups["SUPPORT_DATA"].groups["GEOLOCATIONS"].variables["satellite_altitude"][:][0]
  #get necessary attributes 
  fv=data._FillValue
  
  #get lat and lon information 
  min_lat=np.min(lat)
  max_lat=np.max(lat)
  min_lon=np.min(lon)
  max_lon=np.max(lon)

  # if min_lat>llcrn[1] or max_lat<urcrn[1]:
  #   print("Not enough latitude coverage")
  # elif min_lon>llcrn[0] or max_lon<urcrn[0]:
  #   print("Not enough longitude coverage")
  #   return 0
  # elif (min_lat>llcrn[1] or max_lat<urcrn[1]) and (min_lon>llcrn[0] or max_lon<urcrn[0]):
  #   print("Not enough longitude and latitude coverage")
  #   return 0
  #get the data as an array and mask fill/missing values
  dataArray=np.array(data[0][:][:])
  dataArray[dataArray==fv]=np.nan
  data=dataArray
  
  num_cols = int((urcrn[0]-llcrn[0])//d_lon)
  num_rows = int((urcrn[1]-llcrn[1])//d_lat)
  matrix = np.zeros((num_rows,num_cols))
  alt_matrix = np.zeros((num_rows,num_cols))

  lat_cords = np.linspace(llcrn[1],urcrn[1]+d_lat,num_rows)
  lon_cords = np.linspace(llcrn[0],urcrn[0]+d_lon,num_cols)
  for i, la in enumerate(lat_cords):
    for j, lo in enumerate(lon_cords):
      #calculation to find nearest point in data to entered location (haversine formula)
      R=6371000#radius of the earth in meters
      lat1=np.radians(la)
      lat2=np.radians(lat)
      delta_lat=np.radians(lat-la)
      delta_lon=np.radians(lon-lo)
      a=(np.sin(delta_lat/2))*(np.sin(delta_lat/2))+(np.cos(lat1))*(np.cos(lat2))*(np.sin(delta_lon/2))*(np.sin(delta_lon/2))
      c=2*np.arcsin(np.sqrt(a))
      # d=R*c
      #gets (and then prints) the x,y location of the nearest point in data to entered location, accounting for no data values
      x,y=np.unravel_index(np.nanargmin(c),c.shape)
      alt_matrix[i,j] = alt[x]
      #print(x,y)
      if np.isnan(dataArray[x,y]):
        average = threebythree(dataArray,x,y)
        if average:
          matrix[i,j] = average
        else:
          pass
      elif dataArray[x,y] != fv:
        matrix[i,j] = dataArray[x,y]
  return matrix, alt_matrix, np.meshgrid(lon_cords,lat_cords)

def threebythree(dataArray,x,y):
  if x < 1:
      x+=1
  if x > dataArray.shape[0]-2:
      x-=2
  if y < 1:
      y+=1
  if y > dataArray.shape[1]-2:
      y-=2
  three_by_three=dataArray[x-1:x+2,y-1:y+2]
  nnan=np.count_nonzero(~np.isnan(three_by_three))
  if nnan == 0:
      average = fivebyfive(dataArray,x,y)
      if average:
        return average
      else:
        return None
  else:
      average=np.nanmean(three_by_three)
      return average      

def fivebyfive(dataArray,x,y):
  if x < 2:
      x+=1
  if x > dataArray.shape[0]-3:
      x-=1
  if y < 2:
      y+=1
  if y > dataArray.shape[1]-3:
      y-=1
  five_by_five=dataArray[x-2:x+3,y-2:y+3]
  nnan=np.count_nonzero(~np.isnan(five_by_five))
  if nnan == 0:
      return None
  else:
      average=np.nanmean(five_by_five)
      return average

def avgGrid(FILE_LIST,d_lat=0.025,d_lon=0.025,llcrn=(-10.56, 51.39),urcrn=(-5.34, 55.43)):
  avg_matrix = []
  avg_alt = []
  lon = None
  lat = None
  for FILE_NAME in tqdm(FILE_LIST,"Processing Files: "):
    grid,alt,(lon,lat) = mapGrid(FILE_NAME,d_lat,d_lon,llcrn,urcrn)
    avg_matrix.append(grid)
    avg_alt.append(alt)
  mask = [np.any(mat) for mat in avg_matrix]
  return np.sum(avg_matrix,axis=0)/np.sum(mask,axis=0),np.mean(avg_alt,axis=0),lon,lat

def plotMap(grid,lon,lat,extent):
  
  def image_spoof(self, tile): # this function pretends not to be a Python script
    url = self._image_url(tile) # get the url of the street map API
    req = Request(url) # start request
    req.add_header('User-agent','Anaconda 3') # add user agent to request
    fh = urlopen(req) 
    im_data = io.BytesIO(fh.read()) # get image
    fh.close() # close url
    img = Image.open(im_data) # open image with PIL
    img = img.convert(self.desired_tile_form) # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy

  cimgt.QuadtreeTiles.get_image = image_spoof # reformat web request for street map spoofing
  osm_img = cimgt.QuadtreeTiles() # spoofed, downloaded street map

  fig = plt.figure(figsize=(12,9)) # open matplotlib figure
  ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
  zoom = 0.00075 # for zooming out of center point
  # extent = [np.min(lon),np.max(lon),np.min(lat),np.max(lat)] # adjust to zoom
  #extent = [-6.4,-6.11,53.282,53.4]
  ax1.set_extent(extent) # set extents

  scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2,350.0))) # empirical solve for scale based on zoom
  scale = (scale<20) and scale or 19# scale cannot be larger than 19
  ax1.add_image(osm_img, int(scale))# add OSM with zoom specification
  # NOTE: zoom specifications should be selected based on extent:
  # -- 2     = coarse image, select for worldwide or continental scales
  # -- 4-6   = medium coarseness, select for countries and larger states
  # -- 6-10  = medium fineness, select for smaller states, regions, and cities
  # -- 10-12 = fine image, select for city boundaries and zip codes
  # -- 14+   = extremely fine image, select for roads, blocks, buildings
  my_cmap = plt.cm.get_cmap("jet")
  c = plt.pcolormesh(lon,lat,grid,transform=ccrs.PlateCarree(),cmap=my_cmap,alpha=0.3)
  fig.colorbar(c,extend="both",label="ug/m3")
  plt.show() # show the plot
  return ax1

from IPython.display import clear_output
import matplotlib.pyplot as plt
class loss_plt(tf.keras.callbacks.Callback):
  def on_train_begin(self,logs={}):
    self.losses = []
    self.val_losses =[]
    self.rmse = []
    self.val_rmse =[]
    with open('logs.txt','w') as f:
      f.write('---------------------------Train Logs-------------------------------------')
      f.close()

  def on_epoch_end(self,epoch,logs={}):
    clear_output(wait=True)
    self.val_losses.append(logs.get('val_loss'))
    self.losses.append(logs.get('loss'))

    self.val_rmse.append(logs.get('val_root_mean_squared_error'))
    self.rmse.append(logs.get('root_mean_squared_error'))

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(self.val_losses,color="green",label="val_loss")
    plt.plot(self.losses,color="red",label="loss")
    plt.legend()
    plt.title("loss curve");

    plt.subplot(1,2,2)
    plt.plot(self.val_rmse,color="green",label="val_rmse")
    plt.plot(self.rmse,color="red",label="rmse")
    plt.legend()
    plt.title("rmse curve");
    plt.tight_layout()
    plt.show()
    log_lines = f'epoch {epoch+1}/50 loss: {self.losses[-1]:.03f} rmse: {self.rmse[-1]:.04f} val_loss: {self.val_losses[-1]:.03f} val_rmse:{self.val_rmse[-1]:.04f}'
    print(log_lines)
    with open('logs.txt','a') as f:
      f.write('\n'+log_lines)
      f.close()

def groundGrid(date,asso_df,llcrn,urcrn):
  num_cols = int((urcrn[0]-llcrn[0])//0.01)
  num_rows = int((urcrn[1]-llcrn[1])//0.01)
  lat_cords = np.linspace(llcrn[1],urcrn[1]+0.01,num_rows)
  lon_cords = np.linspace(llcrn[0],urcrn[0]+0.01,num_cols)

  matrix = np.zeros((num_rows,num_cols))
  st_datas = []
  for i in range(12):
    x = pd.read_csv(f'station_{i}.csv')
    st_data = x[x["Date"].apply(getDate)==date]["NO2"].values
    st_data = st_data[0] if len(st_data)>0 else 0
    st_datas.append(st_data)

  for i, la in enumerate(lat_cords):
    for j, lo in enumerate(lon_cords):
      id = int(asso_df[(asso_df["latitude"].astype("float")==la)&(asso_df["longitude"].astype("float")==lo)]["station_id"].values[0].split("_")[-1])
      matrix[i,j] = st_datas[id]
  return matrix,np.meshgrid(lon_cords,lat_cords)