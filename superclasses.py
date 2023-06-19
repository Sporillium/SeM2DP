# File for defining Superclass groupings:
# New Superclass Structure:

#-1
void_class = [-1]
'''Void'''
# Total Weight in Dataset: 0.0725 (7.25%)

#0
walls_class = [0]
'''Walls'''
# Total Weight in Dataset: 0.1576 (15.76%)

#1
buildings_class = [1, 25, 48, 79, 84, 114]
'''Building, House, Skyscraper, Hut, Tower, Tent'''
# Total Weight in Dataset: 0.1172 (11.72%)

#2
sky_class = [2]
'''Sky'''
# Total Weight in Dataset: 0.0878 (8.78%)

#3
natural_terrain_class = [9, 13, 16, 21, 26, 29, 46, 52, 60, 68, 91, 94, 109, 113, 128]
'''Grass, Earth, Mountain, Water, Sea, Field, Sand, Path, River, Hill, Dirt, Land, Swimming Pool, Waterfall, Lake'''
# Total Weight in Dataset: 0.0705 (7.05%)

#4
outdoor_objects_class = [4, 17, 34, 42, 66, 72, 104, 125, 132]
'''Tree, Plant, Rock, Column, Flower, Palm, Fountain, Flowerpot, Sculpture'''
# Total Weight in Dataset: 0.0675 (6.75%)

#5
floors_class = [3, 28]
'''Floor, Carpet'''
# Total Weight in Dataset: 0.0667 (6.67%)

#6
wall_objects_class = [8, 14, 18, 22, 27, 58, 63, 100, 130, 144, 148]
'''Window, Door, Curtain, Painting, Mirror, Screen Door, Blind, Poster, Projection Screen, Notice Board, Clock'''
# Total Weight in Dataset: 0.0583 (5.83%)

#7
artificial_terrain_class = [6, 11, 54]
'''Road, Sidewalk, Runway'''
# Total Weight in Dataset: 0.0581 (5.81%)

#8
ceilings_class = [5, 36, 82, 85, 134]
'''Ceiling, Lamp, Light, Chandelier, Sconce'''
# Total Weight in Dataset: 0.0495 (4.95%)

#9
cabinets_class = [10, 24, 35, 40, 44, 55, 62, 99]
'''Cabinet, Shelf, Wardrobe, Pedestal, Chest of Drawers, Display Case, Bookcase, Sideboard'''
# Total Weight in Dataset: 0.0348 (3.48%)

#10
dynamic_objects_class = [12, 20, 76, 80, 83, 90, 102, 103, 116, 126, 127]
'''Person, Car, Boat, Bus, Truck, Aeroplane, Van, Ship, Motorbike, Animal, Bicycle'''
# Total Weight in Dataset: 0.0316 (3.16%)

#11
tables_class = [15, 33, 41, 45, 56, 64, 67, 70, 73, 77, 88, 98, 108, 111, 112, 119, 120, 135, 137, 142, 147]
'''Table, Desk, Box, Counter, Pool Table, Coffee Table, Book, Countertop, Kitchen Island, Bar, Cubicle, Bottle, Toy,
   Barrel, Basket, Ball, Food, Vase, Tray, Plate, Glass'''
# Total Weight in Dataset: 0.0308 (3.08%)

#12
bed_class = [7, 39, 57, 81, 92, 115, 117, 131]
'''Bed, Cushion, Pillow, Towel, Apparel, Bag, Cradle, Blanket'''
# Total Weight in Dataset: 0.03 (3.00%)

#13
seating_class = [19, 23, 30, 31, 69, 75, 97, 110]
'''Chair, Sofa, Armchair, Seat, Bench, Swivel Chair, Ottoman, Stool'''
# Total Weight in Dataset: 0.029 (2.9%)

#14
large_structures_class = [32, 38, 43, 51, 53, 59, 61, 86, 87, 93, 95, 96, 101, 105, 106, 121, 122, 123, 136, 140, 149]
'''Fence, Railing, Signboard, Grandstand, Stairs, Stariway, Bridge, Awning, Streetlight, Pole, Bannister, Escalator,
   Stage, Conveyor Belt, Canopy, Step, Tank, Marquee, Traffic Light, Pier, Flag'''
# Total Weight in Dataset: 0.021 (2.1%)

#15
appliances_class = [37, 47, 49, 50, 65, 71, 74, 78, 89, 107, 118, 124, 129, 133, 138, 139, 141, 143, 145, 146]
'''Bathtub, Sink, Fireplace, Refrigerator, Toilet, Stove, Computer, Arcade Machine, Television, Washer, Oven,
   Microwave, Dishwasher, Exhaust Hood, Trash Can, Fan, CRT Screen, Monitor, Shower, Radiator'''
# Total Weight in Dataset: 0.0171 (1.71%)

def populate_superclasses(zero_index = True):
    superclasses = {}
    superclasses_initial = {
        -1:void_class,
        0:walls_class,
        1:buildings_class,
        2:sky_class,
        3:natural_terrain_class,
        4:outdoor_objects_class,
        5:floors_class,
        6:wall_objects_class,
        7:artificial_terrain_class,
        8:ceilings_class,
        9:cabinets_class,
        10:dynamic_objects_class,
        11:tables_class,
        12:bed_class,
        13:seating_class,
        14:large_structures_class,
        15:appliances_class
    }
    if not zero_index:
        for k,v in superclasses_initial.items():
            val = [x+1 for x in v]
            superclasses[k] = val
        return superclasses
    else:
        return superclasses_initial
