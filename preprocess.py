import json
import os
import pandas
import re
import sys

re_json = re.compile( r'json$', re.I )
re_eng = re.compile( r'^en_', re.I )

data_dir = "/home/daniel/mnt2/Data/abo-listings/listings/metadata"
img_base_dir = "/home/daniel/mnt2/Data/abo-images-small/images"
img_filepath = os.path.join( img_base_dir, "metadata/images.csv" )
img_dir = os.path.join( img_base_dir, "small" ) 

if os.path.exists( img_filepath ) :
    print( f"--> {img_filepath} exists" )
    images_df = pandas.read_csv( img_filepath )
else :
    print( f"--> {img_filepath} does NOT exist" )
    sys.exit()

json_filenames = list( filter( lambda x : re.search( re_json, x ), os.listdir( data_dir ) ) )

all_data = []
for i, json_filename in enumerate( json_filenames ) :
    print( f"--> [ {i} / {len(json_filenames)} ] - {json_filename}" )
    json_filepath = os.path.join( data_dir, json_filename )
    with open( json_filepath, 'r' ) as f :
        data = f.readlines()

    for line in data :
        json_data = json.loads( line )
        try :
            if re.match( re_eng, json_data[ "brand" ][ 0 ][ "language_tag" ] ) is None :
                continue
        except :
            continue

        temp = {}
        try :
            temp[ "item_id" ] = json_data[ "item_id" ]
            temp[ "main_image_id" ] = json_data[ "main_image_id" ]
            temp[ "item_name" ] = json_data[ "item_name" ][ 0 ][ "value" ]
            temp[ "product_type" ] = json_data[ "product_type" ][ 0 ][ "value" ]
            temp[ "description" ] = "\n".join( [ j[ "value" ] for j in json_data[ "bullet_point" ] ] )
            temp[ "keywords" ] = [ j[ "value" ] for j in json_data[ "item_keywords" ] ] 
            image_location = images_df[ images_df[ "image_id" ] == temp[ "main_image_id" ] ].path.values[ 0 ] 
            temp[ "image_location" ] = os.path.join( img_dir, image_location )
            all_data.append( temp )
        except Exception as e :
#            print( f"--> Error: {str(e)}" )
            pass

with open( "abo_preprocessed.json", 'w' ) as f :
    json.dump( all_data, f )
