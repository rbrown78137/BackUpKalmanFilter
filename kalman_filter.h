#include <Eigen/Dense>
using namespace Eigen;

class ExtendedKalmanFilter{
    private:
    MatrixXd last_predicted_state;
    MatrixXd last_inputs;
    MatrixXd last_prediction_error;
    // X Matrix
    MatrixXd predict_next_state(MatrixXd current_state, MatrixXd inputs, double change_in_time){
        MatrixXd next_state(3,1);
        double V = inputs(1,0); //Velocity 
        double W = inputs(0,0); // Turning Rate
        double P = current_state(0,0); // X Position 
        double Y = current_state(1,0); // Y Position
        double H = current_state(2,0); // Heading
        double A = 0;
        double B = 0;
        double C = 0;
        if(abs(W)<0.00001){
            B = V * change_in_time;
        }else{
            if(abs(V)>0.0001){
                A = (V/W) * (1-cos(W*change_in_time));
                B = (V/W) * (sin(W*change_in_time));
                C = W * change_in_time;
            }
        }
        double new_P = (P-A)*cos(C) - (Y-B)*sin(C);
        double new_Y = (P-A)*sin(C) + (Y-B)*cos(C);
        double new_H = H + W * change_in_time;
        next_state(0,0)  = new_P;
        next_state(1,0)  = new_Y;
        next_state(2,0)  = new_H;
        return next_state;
    }

    // J_x matrix
    MatrixXd prediction_jacobian(MatrixXd inputs, double change_in_time){
        double W = inputs(0,0);
        MatrixXd jacobian(3,3);
        jacobian(0,0) = cos(W*change_in_time);
        jacobian(0,1) = -sin(W*change_in_time);
        jacobian(0,2) = 0;
        jacobian(1,0) = sin(W*change_in_time);
        jacobian(1,1) = cos(W*change_in_time);
        jacobian(1,2) = 0;
        jacobian(2,0) = 0;
        jacobian(2,1) = 0;
        jacobian(2,2) = 1;
        return jacobian;
    }

    // R Matrix
    MatrixXd sensor_noise_matrix(MatrixXd sensor_uncertainty){
        MatrixXd sensor_noise = MatrixXd::Zero(3,3);
        sensor_noise(0,0) = sensor_uncertainty(0,0);
        sensor_noise(1,1) = sensor_uncertainty(1,0);
        sensor_noise(2,2) = sensor_uncertainty(2,0);
        return sensor_noise;
    }

    // Q Matrix
    MatrixXd process_noise_matrix(double change_in_time){
        double A = 1.7; // Max acceleration of car in M/S^2
        double max_speed = 7.0;
        double max_steering_angle = M_PI / 6;
        double car_length = 0.3;
        double max_turning_rate = max_speed/sqrt(pow(car_length,2)+(pow(car_length/2,2))*pow(cos(max_steering_angle)/sin(max_steering_angle),2));
        MatrixXd process_noise = MatrixXd::Zero(3,3);
        process_noise(0,0) = 0.5 * A * change_in_time * change_in_time; // ADD P AND Y LATER IF NEEDED 
        process_noise(1,1) = 0.5 * A * change_in_time * change_in_time; // ADD P AND Y LATER IF NEEDED 
        process_noise(2,2) = max_turning_rate * change_in_time;
        return process_noise;
    }

    // H Matrix
    MatrixXd observation_matrix(){
        return MatrixXd::Identity(3,3);
    }

    // J_h Matrix
    MatrixXd observation_jacobian_matrix(){
        return MatrixXd::Identity(3,3);
    }

    // General Extended Kalman Filter Equations
    MatrixXd predict_next_prediction_error(MatrixXd previous_prediction_error, MatrixXd jacobian_prediction, MatrixXd process_noise){
        return jacobian_prediction * previous_prediction_error * jacobian_prediction.transpose() + process_noise;

    }

    MatrixXd kalman_gain(MatrixXd prediction_error, MatrixXd jacobian_observation, MatrixXd sensor_noise){
        return prediction_error * jacobian_observation.transpose() * (jacobian_observation * prediction_error * jacobian_observation.transpose() + sensor_noise).inverse();
    }

    MatrixXd update_process_noise(MatrixXd previous_prediction_error, MatrixXd jacobian_observation, MatrixXd sensor_noise){
        MatrixXd kalman_gain_matrix = kalman_gain(previous_prediction_error, jacobian_observation, sensor_noise);
        MatrixXd identity = MatrixXd::Identity(previous_prediction_error.rows(),previous_prediction_error.rows());
        return (identity- (kalman_gain_matrix * jacobian_observation)) * previous_prediction_error;
    }

    MatrixXd update_prediction(MatrixXd previous_prediction, MatrixXd sensor_observation, MatrixXd observation_matrix, MatrixXd previous_prediction_error, MatrixXd jacobian_observation, MatrixXd sensor_noise){
        MatrixXd kalman_gain_matrix = kalman_gain(previous_prediction_error, jacobian_observation, sensor_noise);
        return previous_prediction + kalman_gain_matrix * (sensor_observation - observation_matrix * previous_prediction);
    }
    
    public:
    ExtendedKalmanFilter(double P, double Y, double H, double V, double W, double P_uncertainty, double Y_uncertainty, double H_uncertainty){
        this->last_predicted_state = MatrixXd(3,1);
        this->last_predicted_state(0,0) = P;
        this->last_predicted_state(1,0) = Y;
        this->last_predicted_state(2,0) = H;
        this->last_inputs = MatrixXd(2,1);
        this->last_inputs(0,0) = W;
        this->last_inputs(1,0) = V;
        this->last_prediction_error = MatrixXd(3,3);
        this->last_prediction_error(0,0) = P_uncertainty;
        this->last_prediction_error(1,1) = Y_uncertainty;
        this->last_prediction_error(2,2) = H_uncertainty;
    }

    ExtendedKalmanFilter(MatrixXd initial_prediction, MatrixXd initial_inputs, MatrixXd initial_error){
        this->last_predicted_state = initial_prediction;
        this->last_inputs = initial_inputs;
        this->last_prediction_error = initial_error;
    }

    MatrixXd get_next_prediction(MatrixXd sensor_observation,MatrixXd sensor_uncertainty, MatrixXd new_inputs, double change_in_time){
        MatrixXd Q = process_noise_matrix(change_in_time);
        MatrixXd R = this->sensor_noise_matrix(sensor_uncertainty);
        MatrixXd X_hat = this->predict_next_state(this->last_predicted_state,this->last_inputs,change_in_time);
        MatrixXd J_X = this->prediction_jacobian(this->last_inputs,change_in_time);
        MatrixXd P_hat = predict_next_prediction_error(this->last_prediction_error,J_X,Q);
        MatrixXd X = this->update_prediction(X_hat,sensor_observation,this->observation_matrix(),P_hat,this->observation_jacobian_matrix(),R);
        this->last_predicted_state = X;
        MatrixXd P = this->update_process_noise(P_hat,this->observation_jacobian_matrix(),R);
        this->last_prediction_error = P;
        this->last_inputs = new_inputs;
        return this->last_predicted_state;
    }
    
    MatrixXd get_next_prediction(double P,double Y, double H, double sigma_P, double sigma_Y, double sigma_H, double W, double V, double change_in_time){
        MatrixXd sensor_observation(3,1);
        sensor_observation(0,0) = P;
        sensor_observation(1,0) = Y;
        sensor_observation(2,0) = H;
        MatrixXd sensor_uncertainty(3,1);
        sensor_uncertainty(0,0) = sigma_P;
        sensor_uncertainty(1,0) = sigma_Y;
        sensor_uncertainty(2,0) = sigma_H;
        MatrixXd new_inputs(2,1);
        new_inputs(0,0) = W;
        new_inputs(1,0) = V;
        return this->get_next_prediction(sensor_observation, sensor_uncertainty, new_inputs, change_in_time);
    }
    
    MatrixXd get_next_prediction(double W, double V, double change_in_time){
        MatrixXd new_inputs(2,1);
        new_inputs(0,0) = W;
        new_inputs(1,0) = V;
        return this->get_next_prediction(new_inputs,change_in_time);
    }
    
    MatrixXd get_next_prediction(MatrixXd new_inputs, double change_in_time){
        MatrixXd Q = process_noise_matrix(change_in_time);
        MatrixXd X_hat = this->predict_next_state(this->last_predicted_state,this->last_inputs,change_in_time);
        MatrixXd J_X = this->prediction_jacobian(this->last_inputs,change_in_time);
        MatrixXd P_hat = predict_next_prediction_error(this->last_prediction_error,J_X,Q);
        this->last_predicted_state = X_hat;
        this->last_prediction_error = P_hat;
        this->last_inputs = new_inputs;
        return this->last_predicted_state;
    }
    
    MatrixXd get_next_prediction(double change_in_time){
        MatrixXd Q = process_noise_matrix(change_in_time);
        MatrixXd X_hat = this->predict_next_state(this->last_predicted_state,this->last_inputs,change_in_time);
        MatrixXd J_X = this->prediction_jacobian(this->last_inputs,change_in_time);
        MatrixXd P_hat = predict_next_prediction_error(this->last_prediction_error,J_X,Q);
        this->last_predicted_state = X_hat;
        this->last_prediction_error = P_hat;
        return this->last_predicted_state;
    }
    
    MatrixXd get_current_state(){
        return this->last_predicted_state;
    }
};
