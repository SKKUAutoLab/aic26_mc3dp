# ==================================================================== #
# Copyright (C) 2022 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ==================================================================== #
# from __future__ import annotations

from typing import List

from core.objects.instance import Instance
from core.utils.constants import AppleRGB
from core.objects.general_object import GeneralObject
from core.objects.moving_model import(
	MovingModel,
	MovingState
)
from core.objects.driver_model import MotorbikeDriverModel
from core.objects.motion_model import MotionModel

# MARK: - General_Moving_Object (General Moving Object)

class General_Moving_Object(GeneralObject, MotionModel):
	# MARK: Class Property
	
	min_entering_distance: int = 0
	min_traveled_distance: int = 100
	min_hit_streak       : int = 10
	max_age              : int = 1
	
	# MARK: Magic Functions
	
	def __init__(self, **kwargs):
		# NOTE: For matching, flow estimation
		GeneralObject.__init__(self, **kwargs)
		MotionModel.__init__(self, **kwargs)

	# MARK: Configure
	
	@classmethod
	def gmo_from_detection(cls, detection: Instance, **kwargs):
		"""Create the new class/object"""
		return cls(
			frame_index = detection.frame_index,
			bbox        = detection.bbox,
			polygon     = detection.polygon,
			confidence  = detection.confidence,
			label       = detection.label,
			roi_uuid    = detection.roi_uuid,
			**kwargs
		)
		
	# MARK: Update
	
	def update_gmo(self, detection: Instance):
		"""Main function for update all the general moving object

		Args:
			detection (Instance):
				Detection from the detector
		"""
		# NOTE: First, update ``GeneralObject``
		self.update_go_from_detection(instance=detection)
		
		# NOTE: Second, update motion model
		self.update_motion_state()
		
	# MARK: Visualize

	def draw(self, drawing, **kwargs):
		""" Draw the object base one the moving state of its.

		Args:
			drawing:
				image for drawing

		"""
		if self.is_confirmed:
			GeneralObject.draw(self, drawing=drawing, label=True, color=AppleRGB.values()[0], **kwargs)
		elif self.is_counting:
			GeneralObject.draw(self, drawing=drawing, label=True, color=AppleRGB.values()[0], **kwargs)
		elif self.is_counted:
			GeneralObject.draw(self, drawing=drawing, label=True, trajectory=True, color=AppleRGB.values()[self.moi_uuid], **kwargs)
		elif self.is_exiting:
			GeneralObject.draw(self, drawing=drawing, label=True, trajectory=True, color=AppleRGB.values()[self.moi_uuid], **kwargs)
