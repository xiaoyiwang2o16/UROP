import glob
import csv
import numpy as np
import math 





def get_mean(reader):
	mean = [0]*344
	#for line in reader:
	n = len(reader)
	for temp in reader:
		for i in range(len(temp)):
			mean[i] += temp[i]/n
	return mean 

def get_sd(reader, mean):
	sd = [0]*344
	n = len(reader)
	for temp in reader:
		for i in range(len(temp)):
			sd[i] += (temp[i]-mean[i])**2
	for dev in range(len(sd)):
		sd[dev] = math.sqrt(sd[dev]/(n-1))
	return sd

def normalize(reader, mean, sd):
	new = []
	for temp in reader:
		row = []
		for i in range(len(temp)):
			row.append((temp[i]-mean[i])/sd[i])
		new.append(row)
	return new


def mod_file():
	file = glob.glob('25day_users.csv')
	with open(file[0], 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=" ", quotechar= '|')

		new = ['user_id,survey_Presleep_media_interation,survey_Presleep_personal_interaction,survey_Time_in_bed,survey_academic_duration,survey_awakening_duration,survey_awakening_occations,survey_caffeine_count,survey_count_academic,survey_count_awakening,survey_count_extracurricular,survey_drugs,survey_drugs_alcohol,survey_drugs_alert,survey_drugs_sleepy,survey_drugs_tired,survey_exercise_duration,survey_exercise_occations,survey_extracurricular_duration,survey_negative_interaction,survey_no_sleep_24,survey_overslept,survey_positive_interaction,survey_pre_sleep_activity,survey_sleep_latency,survey_sleep_try_time_mins_since_midnight,survey_study_duration,survey_wake_reason,survey_wake_time_mins_since_midnight,survey_count_nap,survey_nap_duration,survey_nap_occations,survey_survey_first_event_0H-3H,survey_survey_first_event_3H-10H,survey_survey_first_event_10H-17H,survey_survey_first_event_17H-24H,survey_survey_caffeine_0H-3H,survey_survey_caffeine_3H-10H,survey_survey_caffeine_10H-17H,survey_survey_caffeine_17H-24H,phys_0H-3H:sumAUC,phys_0H-3H:sumAUCFull,phys_0H-3H:medianRiseTime,phys_0H-3H:medianAmplitude,phys_0H-3H:countPeaks,phys_0H-3H:sdPeaks30min,phys_0H-3H:medPeaks30min,phys_0H-3H:percentMedPeak,phys_0H-3H:percentHighPeak,phys_0H-3H:sumAUCNoArtifact,phys_0H-3H:sumAUCFullNoArtifact,phys_0H-3H:medianRiseTimeNoArtifact,phys_0H-3H:medianAmplitudeNoArtifact,phys_0H-3H:countPeaksNoArtifact,phys_0H-3H:sdPeaks30minNoArtifact,phys_0H-3H:medPeaks30minNoArtifact,phys_0H-3H:percentMedPeakNoArtifact,phys_0H-3H:percentHighPeakNoArtifact,phys_0H-3H:sclPercentOff,phys_0H-3H:sclMaxUnnorm,phys_0H-3H:sclMedUnnorm,phys_0H-3H:sclMeanUnnorm,phys_0H-3H:sclMedianNorm,phys_0H-3H:sclSDnorm,phys_0H-3H:sclMeanDeriv,phys_0H-3H:stepCount,phys_0H-3H:meanMovementStepTime,phys_0H-3H:stillnessPercent,phys_0H-3H:sumStillnessWeightedAUC,phys_0H-3H:sumStepsWeightedAUC,phys_0H-3H:sumStillnessWeightedPeaks,phys_0H-3H:maxStillnessWeightedPeaks,phys_0H-3H:sumStepsWeightedPeaks,phys_0H-3H:medStepsWeightedPeaks,phys_0H-3H:sumTempWeightedAUC,phys_0H-3H:sumTempWeightedPeaks,phys_0H-3H:maxTempWeightedPeaks,phys_0H-3H:maxRawTemp,phys_0H-3H:minRawTemp,phys_0H-3H:sdRawTemp,phys_0H-3H:medRawTemp,phys_0H-3H:sdStillnessTemp,phys_0H-3H:medStillnessTemp,phys_3H-10H:sumAUC,phys_3H-10H:sumAUCFull,phys_3H-10H:medianRiseTime,phys_3H-10H:medianAmplitude,phys_3H-10H:countPeaks,phys_3H-10H:sdPeaks30min,phys_3H-10H:medPeaks30min,phys_3H-10H:percentMedPeak,phys_3H-10H:percentHighPeak,phys_3H-10H:sumAUCNoArtifact,phys_3H-10H:sumAUCFullNoArtifact,phys_3H-10H:medianRiseTimeNoArtifact,phys_3H-10H:medianAmplitudeNoArtifact,phys_3H-10H:countPeaksNoArtifact,phys_3H-10H:sdPeaks30minNoArtifact,phys_3H-10H:medPeaks30minNoArtifact,phys_3H-10H:percentMedPeakNoArtifact,phys_3H-10H:percentHighPeakNoArtifact,phys_3H-10H:sclPercentOff,phys_3H-10H:sclMaxUnnorm,phys_3H-10H:sclMedUnnorm,phys_3H-10H:sclMeanUnnorm,phys_3H-10H:sclMedianNorm,phys_3H-10H:sclSDnorm,phys_3H-10H:sclMeanDeriv,phys_3H-10H:stepCount,phys_3H-10H:meanMovementStepTime,phys_3H-10H:stillnessPercent,phys_3H-10H:sumStillnessWeightedAUC,phys_3H-10H:sumStepsWeightedAUC,phys_3H-10H:sumStillnessWeightedPeaks,phys_3H-10H:maxStillnessWeightedPeaks,phys_3H-10H:sumStepsWeightedPeaks,phys_3H-10H:medStepsWeightedPeaks,phys_3H-10H:sumTempWeightedAUC,phys_3H-10H:sumTempWeightedPeaks,phys_3H-10H:maxTempWeightedPeaks,phys_3H-10H:maxRawTemp,phys_3H-10H:minRawTemp,phys_3H-10H:sdRawTemp,phys_3H-10H:medRawTemp,phys_3H-10H:sdStillnessTemp,phys_3H-10H:medStillnessTemp,phys_10H-17H:sumAUC,phys_10H-17H:sumAUCFull,phys_10H-17H:medianRiseTime,phys_10H-17H:medianAmplitude,phys_10H-17H:countPeaks,phys_10H-17H:sdPeaks30min,phys_10H-17H:medPeaks30min,phys_10H-17H:percentMedPeak,phys_10H-17H:percentHighPeak,phys_10H-17H:sumAUCNoArtifact,phys_10H-17H:sumAUCFullNoArtifact,phys_10H-17H:medianRiseTimeNoArtifact,phys_10H-17H:medianAmplitudeNoArtifact,phys_10H-17H:countPeaksNoArtifact,phys_10H-17H:sdPeaks30minNoArtifact,phys_10H-17H:medPeaks30minNoArtifact,phys_10H-17H:percentMedPeakNoArtifact,phys_10H-17H:percentHighPeakNoArtifact,phys_10H-17H:sclPercentOff,phys_10H-17H:sclMaxUnnorm,phys_10H-17H:sclMedUnnorm,phys_10H-17H:sclMeanUnnorm,phys_10H-17H:sclMedianNorm,phys_10H-17H:sclSDnorm,phys_10H-17H:sclMeanDeriv,phys_10H-17H:stepCount,phys_10H-17H:meanMovementStepTime,phys_10H-17H:stillnessPercent,phys_10H-17H:sumStillnessWeightedAUC,phys_10H-17H:sumStepsWeightedAUC,phys_10H-17H:sumStillnessWeightedPeaks,phys_10H-17H:maxStillnessWeightedPeaks,phys_10H-17H:sumStepsWeightedPeaks,phys_10H-17H:medStepsWeightedPeaks,phys_10H-17H:sumTempWeightedAUC,phys_10H-17H:sumTempWeightedPeaks,phys_10H-17H:maxTempWeightedPeaks,phys_10H-17H:maxRawTemp,phys_10H-17H:minRawTemp,phys_10H-17H:sdRawTemp,phys_10H-17H:medRawTemp,phys_10H-17H:sdStillnessTemp,phys_10H-17H:medStillnessTemp,phys_17H+:sumAUC,phys_17H+:sumAUCFull,phys_17H+:medianRiseTime,phys_17H+:medianAmplitude,phys_17H+:countPeaks,phys_17H+:sdPeaks30min,phys_17H+:medPeaks30min,phys_17H+:percentMedPeak,phys_17H+:percentHighPeak,phys_17H+:sumAUCNoArtifact,phys_17H+:sumAUCFullNoArtifact,phys_17H+:medianRiseTimeNoArtifact,phys_17H+:medianAmplitudeNoArtifact,phys_17H+:countPeaksNoArtifact,phys_17H+:sdPeaks30minNoArtifact,phys_17H+:medPeaks30minNoArtifact,phys_17H+:percentMedPeakNoArtifact,phys_17H+:percentHighPeakNoArtifact,phys_17H+:sclPercentOff,phys_17H+:sclMaxUnnorm,phys_17H+:sclMedUnnorm,phys_17H+:sclMeanUnnorm,phys_17H+:sclMedianNorm,phys_17H+:sclSDnorm,phys_17H+:sclMeanDeriv,phys_17H+:stepCount,phys_17H+:meanMovementStepTime,phys_17H+:stillnessPercent,phys_17H+:sumStillnessWeightedAUC,phys_17H+:sumStepsWeightedAUC,phys_17H+:sumStillnessWeightedPeaks,phys_17H+:maxStillnessWeightedPeaks,phys_17H+:sumStepsWeightedPeaks,phys_17H+:medStepsWeightedPeaks,phys_17H+:sumTempWeightedAUC,phys_17H+:sumTempWeightedPeaks,phys_17H+:maxTempWeightedPeaks,phys_17H+:maxRawTemp,phys_17H+:minRawTemp,phys_17H+:sdRawTemp,phys_17H+:medRawTemp,phys_17H+:sdStillnessTemp,phys_17H+:medStillnessTemp,location_5_minutes_distance_max,location_5_minutes_distance_mean,location_5_minutes_distance_median,location_5_minutes_distance_std,location_aic_score,location_bic_score,location_enclosing_circle_center_x,location_enclosing_circle_center_y,location_enclosing_circle_radius,location_log_likelihood_of_day,location_number_of_ROIs_visited,location_time_spent_on_campus,location_time_with_indoor_indication,location_time_with_outdoor_indication,location_total_distance_a_day,call_0H-24H_mean_duration,call_0H-24H_median_duration,call_0H-24H_stdev_duration,call_0H-24H_total_duration,call_0H-24H_total_num,call_0H-24H_unique_num,call_0H-24H_mean_duration_incoming,call_0H-24H_median_duration_incoming,call_0H-24H_stdev_duration_incoming,call_0H-24H_total_duration_incoming,call_0H-24H_total_num_incoming,call_0H-24H_unique_num_incoming,call_0H-24H_mean_duration_outgoing,call_0H-24H_median_duration_outgoing,call_0H-24H_stdev_duration_outgoing,call_0H-24H_total_duration_outgoing,call_0H-24H_total_num_outgoing,call_0H-24H_unique_num_outgoing,call_0H-24H_total_num_missed,call_0H-24H_unique_num_missed,sms_0H-24H_total_num,sms_0H-24H_unique_num,sms_0H-24H_total_num_incoming,sms_0H-24H_unique_num_incoming,sms_0H-24H_total_num_outgoing,sms_0H-24H_unique_num_outgoing,sms_0H-3H_total_num,sms_0H-3H_unique_num,sms_0H-3H_total_num_incoming,sms_0H-3H_unique_num_incoming,sms_0H-3H_total_num_outgoing,sms_0H-3H_unique_num_outgoing,sms_17H-24H_total_num,sms_17H-24H_unique_num,sms_17H-24H_total_num_incoming,sms_17H-24H_unique_num_incoming,sms_17H-24H_total_num_outgoing,sms_17H-24H_unique_num_outgoing,sms_10H-17H_total_num,sms_10H-17H_unique_num,sms_10H-17H_total_num_incoming,sms_10H-17H_unique_num_incoming,sms_10H-17H_total_num_outgoing,sms_10H-17H_unique_num_outgoing,sms_3H-10H_total_num,sms_3H-10H_unique_num,sms_3H-10H_total_num_incoming,sms_3H-10H_unique_num_incoming,sms_3H-10H_total_num_outgoing,sms_3H-10H_unique_num_outgoing,screen_0H-24H_mean_duration,screen_0H-24H_median_duration,screen_0H-24H_stdev_duration,screen_0H-24H_total_duration,screen_0H-24H_total_num,screen_0H-3H_mean_duration,screen_0H-3H_median_duration,screen_0H-3H_stdev_duration,screen_0H-3H_total_duration,screen_0H-3H_total_num,screen_17H-24H_mean_duration,screen_17H-24H_median_duration,screen_17H-24H_stdev_duration,screen_17H-24H_total_duration,screen_17H-24H_total_num,screen_10H-17H_mean_duration,screen_10H-17H_median_duration,screen_10H-17H_stdev_duration,screen_10H-17H_total_duration,screen_10H-17H_total_num,screen_3H-10H_mean_duration,screen_3H-10H_median_duration,screen_3H-10H_stdev_duration,screen_3H-10H_total_duration,screen_3H-10H_total_num,weather_sunrise,weather_moon_phase,weather_temperature_max,weather_temperature_min,weather_avg_cloud_cover,weather_avg_dew_point,weather_avg_humidity,weather_avg_pressure,weather_morning_pressure_change,weather_evening_pressure_change,weather_avg_visibility,weather_precip_probability,weather_temperature_rolling_mean,weather_temperature_rolling_std,weather_temperature_today_vs_avg_past,weather_apparentTemperature_rolling_mean,weather_apparentTemperature_rolling_std,weather_apparentTemperature_today_vs_avg_past,weather_pressure_rolling_mean,weather_pressure_rolling_std,weather_pressure_today_vs_avg_past,weather_cloudCover_rolling_mean,weather_cloudCover_rolling_std,weather_cloudCover_today_vs_avg_past,weather_humidity_rolling_mean,weather_humidity_rolling_std,weather_humidity_today_vs_avg_past,weather_windSpeed_rolling_mean,weather_windSpeed_rolling_std,weather_windSpeed_today_vs_avg_past,weather_precipProbability_rolling_mean,weather_precipProbability_rolling_std,weather_precipProbability_today_vs_avg_past,weather_sunlight,weather_quality_of_day,weather_precipType,weather_avg_quality_of_day,weather_max_precip_intensity,weather_median_wind_speed,weather_median_wind_bearing,classifier_friendly_ppt_id,classifier_friendly_day_of_week,classifier_friendly_school_night,label_tomorrow_Alertness_Evening,label_tomorrow_Happiness_Evening,label_tomorrow_Energy_Evening,label_tomorrow_Health_Evening,label_tomorrow_Calmness_Evening,dataset'.split(',')]
		reader = list(reader)[1:]
		
		mmod = []
		labels = []
		values = []
		for line in reader:
			temp = list(map(float, line[1].split(',')[1:-6]))
			tempL = list(map(float, line[1].split(',')[-6:-1]))
			mmod.append(temp)
			labels.append(tempL)
			if line[1].split(',')[-1] == "Val":
				values.append(0)
			elif line[1].split(',')[-1] == "Train":
				values.append(1)
			else:
				values.append(2)
			#values.append(line[1].split(',')[-1])

		mean = get_mean(mmod)
		sd = get_sd(mmod, mean)
		normalized = normalize(mmod, mean, sd)
		for i in range(len(normalized)):
			row = []
			row.append(reader[i][0].split(',')[1][-2:])
			for num in normalized[i]:
				row.append(str(num))
			for label in labels[i]:
				row.append(str(label/100))
			row.append(values[i])
			new.append(row)





	with open('modified_2.csv', 'w') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',')

		for row in new:
			filewriter.writerow(row)
mod_file()




