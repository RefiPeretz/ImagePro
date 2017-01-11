def calc_max_min_corners(ims,Hs):
	p = np.empty((0,2))
	for i in range(len(Hs)):
		s = ims[i].shape
		corners_pos = np.array([[0,0],[s[0]-1,0],[0,s[1]-1],[s[0]-1,s[1]-1]])
		p = np.append(p,apply_homography(corners_pos, Hs[i]),axis = 0)
	min_cor = np.array([np.min(p[:,0]),np.min(p[:,1])])
	max_cor = np.array([np.max(p[:,0]),np.max(p[:,1])])
	return 	min_cor, max_cor