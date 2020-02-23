#pragma once
#include <GL/gl.h>
#include <GL/glu.h>
#include <vector>
#include"fssimplewindow.h"
#include"reeds_shepp.h"
#include <torch/script.h>

struct car {
	int xpos;
	int ypos;
	int heading;
	int velocity;

	car() {

	}
	car(int xpos, int ypos, int heading, int velocity) {
		this->xpos = xpos;
		this->ypos = ypos;
		this->heading = heading;
		this->velocity = velocity;
	}
};

using namespace std;

class CharBitmap
{
public:
	int width, height;
	int scale=4;
	char *pixels;
	char *trans_pixels;
	double ray_total = 0;
	int ray_call_count = 0;


	//creates the bitmap
	void create(int w, int h);

	//sets the value of pixels in the map
	void setPixel(int x, int y, unsigned char p);

	//draws the bitmap on screen
	void draw() const;

	//draws the transformed map on screen
	void drawTransform() const;

	// saves the bitmap in a text file
	void save(const string fName) const;
	void cleanUp();

	// loads the map from text file
	void load(const string fName);

	// draws the car on the screen
	void DrawCar(int, int, int,string);
	bool getPixel(int x, int y);
	void setTransPixel(int x, int y, unsigned char p);

	// draws the generated path on the screen
	void DrawTrajectory(vector<config>*,int, int, int ,int);

	torch::Tensor dist_nearest_obst(config, int,int);	

	torch::Tensor ray_tracing_2(config curr_pos, int front_sweep_angle, int front_step, int back_sweep_angle, int back_step);
	//checks if the map free
	bool isFree(int x, int y);

	bool isBlocked(int x, int y);

	//checks if the transformed map is free
	bool isTransFree(int x, int y);
	CharBitmap();
	~CharBitmap();
};

