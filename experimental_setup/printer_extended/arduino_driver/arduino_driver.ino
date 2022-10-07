#include <Servo.h>

const int BAUD_RATE = 9600;

const int PINS [] = {9, 10}; 

Servo servos[2];

int calibrationA1(int p){
 int bottom_offset = 15;
 int top_offset = 8;
 int range = 180;
 return int(p * ((range - (bottom_offset + top_offset))/(float)range)  + bottom_offset);
}

int calibrationA2(int p){
 int bottom_offset = 10;
 int top_offset = -15;
 int range = 180;
 return int(p * ((range - (bottom_offset + top_offset))/(float)range)  + bottom_offset);
}


typedef int (*CalibrationFn) (int p);
CalibrationFn calibrations [] = {
  calibrationA1,
  calibrationA2
};

void setup() {

  Serial.begin(BAUD_RATE);
  for ( int i = 0 ; i < 2 ; i++ ) {
    servos[i].attach(PINS[i]);
  }
  
  Serial.println("Hi!");
}

void loop() {
  if (Serial.available() > 0) {
  
    for ( int i = 0 ; i < 2 ; i++ ) {
        String posStr = Serial.readStringUntil(';');
        int pos = posStr.toInt();
        int calibrated_pos = calibrations[i](pos);
        servos[i].write(calibrated_pos);
        Serial.println("> a" + String(i+1) +  ": " +  String(pos) + " calibrated to " + String(calibrated_pos));
      
    }
    
    Serial.readString();  
    delay(1000);
    Serial.println("ok.");

  }
}
