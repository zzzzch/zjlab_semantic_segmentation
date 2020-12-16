#pragma once

#include "sgm_types.h"
#include <vector>

class SemiGlobalMatching
{
public:
	SemiGlobalMatching();
	~SemiGlobalMatching();


	enum CensusSize {
		Census5x5 = 0,
		Census9x7
	};

	struct SGMOption {
		uint8	num_paths;			// �ۺ�·���� 4 and 8
		sint32  min_disparity;		// ��С�Ӳ�
		sint32	max_disparity;		// ����Ӳ�

		CensusSize census_size;		// census���ڳߴ�

		bool	is_check_unique;	// �Ƿ���Ψһ��
		float32	uniqueness_ratio;	// Ψһ��Լ����ֵ ����С����-����С����)/��С���� > ��ֵ Ϊ��Ч����

		bool	is_check_lr;		// �Ƿ�������һ����
		float32	lrcheck_thres;		// ����һ����Լ����ֵ

		bool	is_remove_speckles;	// �Ƿ��Ƴ�С����ͨ��
		int		min_speckle_aera;	// ��С����ͨ���������������

		bool	is_fill_holes;		// �Ƿ�����Ӳ�ն�

		// P1,P2 
		// P2 = P2_init / (Ip-Iq)
		sint32  p1;				// �ͷ������P1
		sint32  p2_init;		// �ͷ������P2

		SGMOption(): num_paths(8), min_disparity(0), max_disparity(64), census_size(Census5x5),
		             is_check_unique(true), uniqueness_ratio(0.95f),
		             is_check_lr(true), lrcheck_thres(1.0f),
		             is_remove_speckles(true), min_speckle_aera(20),
		             is_fill_holes(true),
		             p1(10), p2_init(150) { }
	};
public:
	bool Initialize(const sint32& width, const sint32& height, const SGMOption& option);

	bool Match(const uint8* img_left, const uint8* img_right, float32* disp_left);

	bool Reset(const uint32& width, const uint32& height, const SGMOption& option);

private:

	void CensusTransform() const;

	void ComputeCost() const;

	void CostAggregation() const;

	void ComputeDisparity() const;

	void ComputeDisparityRight() const;

	void LRCheck();

	void FillHolesInDispMap();

	void Release();

private:
	SGMOption option_;

	sint32 width_;

	sint32 height_;

	const uint8* img_left_;

	const uint8* img_right_;
	
	void* census_left_;
	
	void* census_right_;
	
	uint8* cost_init_;
	
	uint16* cost_aggr_;

	uint8* cost_aggr_1_;
	uint8* cost_aggr_2_;
	uint8* cost_aggr_3_;
	uint8* cost_aggr_4_;
	uint8* cost_aggr_5_;
	uint8* cost_aggr_6_;
	uint8* cost_aggr_7_;
	uint8* cost_aggr_8_;

	float32* disp_left_;
	float32* disp_right_;

	bool is_initialized_;

	std::vector<std::pair<int, int>> occlusions_;
	std::vector<std::pair<int, int>> mismatches_;
};

