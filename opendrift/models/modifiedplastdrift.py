# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway

import numpy as np
import scipy
import logging; logger = logging.getLogger(__name__)
from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray
from opendrift.config import CONFIG_LEVEL_ESSENTIAL, CONFIG_LEVEL_BASIC, CONFIG_LEVEL_ADVANCED
from shoreline_interactions import one_timestep_constant_p, one_timestep_varying_p
import pyproj
from enum import Enum 
import sys
import functools
from typing import Union, List
from opendrift.errors import WrongMode

Mode = Enum('Mode', ['Config', 'Ready', 'Run', 'Result'])

class PlasticObject(Lagrangian3DArray):
    variables = Lagrangian3DArray.add_variables([
        ('terminal_velocity', {'dtype': np.float32,
                               'units': 'm/s',
                               'level': CONFIG_LEVEL_ESSENTIAL,
            'description': 'Positive value means rising particles (positive buoyancy)',
                               'default': 0.01}),
        ('beached', {'dtype': np.int32,
                    'units': '1',
                    'default': 0}),
        ('height_on_beach', {'dtype': np.float32,
                            'units': 'm',
                            'default': 0}),
        ('last_floating_lon', {'dtype': np.float32,
                               'units': 'm',
                               'default': 0}),
        ('last_floating_lat', {'dtype': np.float32,
                               'units': 'm',
                               'default': 0})
                            ],)


class ModifiedPlastDrift(OceanDrift):
    """Trajectory model based on the OpenDrift framework.

    Propagation of plastics particles with ocean currents and
    additional Stokes drift and wind drag.

    Developed at MET Norway.

    """

    ElementType = PlasticObject

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'sea_surface_height': {'fallback': 0},
        'sea_surface_wave_stokes_drift_x_velocity': {'fallback': 0},
        'sea_surface_wave_stokes_drift_y_velocity': {'fallback': 0},
        'sea_surface_wave_significant_height': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'ocean_vertical_diffusivity': {'fallback': 0.02, 'profiles': True},
        'ocean_mixed_layer_thickness': {'fallback': 50},
        'sea_floor_depth_below_sea_level': {'fallback': 10000},
        'land_binary_mask': {'fallback': None},
        'sea_surface_wave_significant_height': {'fallback': 0},
        'sea_surface_wave_period_at_variance_spectral_density_maximum': {'fallback': 0},
        'beach_angle': {'fallback': np.radians(5)}
        }


    def __init__(self, *args, **kwargs):

        # Call parent constructor
        super(ModifiedPlastDrift, self).__init__(*args, **kwargs)

        # Add specific config settings
        self._add_config({
            # TODO: this option should be moved to OceanDrift
            'vertical_mixing:mixingmodel': {'type': 'enum',
                'enum': ['randomwalk', 'analytical'], 'default': 'analytical',
                'level': CONFIG_LEVEL_ADVANCED, 'description':
                    'Scheme to be used for vertical turbulent mixing'},
            })

        self._set_config_default('drift:vertical_mixing', True)
        self._set_config_default('drift:vertical_advection', True)
        self._set_config_default('drift:use_tabularised_stokes_drift', True)
        self._set_config_default('general:coastline_action', 'beachingmodel')
        self._set_config_default('vertical_mixing:diffusivitymodel', 'windspeed_Sundby1983')

    def update(self):
        """Update positions and properties of elements."""

        # Simply move particles with ambient current
        self.advect_ocean_current()

        #self.update_particle_depth()

        # Advect particles due to Stokes drift
        #self.stokes_drift()

        # Advect particles due to wind-induced shear near surface
        #self.advect_wind()

        self.beaching_resuspension()

    def update_particle_depth(self):

        if self.get_config('drift:vertical_mixing') is True:

            if self.get_config('vertical_mixing:mixingmodel') == 'randomwalk':
                logger.debug('Turbulent mixing of particles using random walk')
                self.vertical_mixing()

            if self.get_config('vertical_mixing:mixingmodel') == 'analytical':
                logger.debug('Submerging according to wind')
                self.elements.z = -np.random.exponential(
                    scale=self.environment.ocean_vertical_diffusivity/
                            self.elements.terminal_velocity,
                    size=self.num_elements_active())

    def beaching_resuspension(self):

        on_land = self.elements.beached == 1
        floating = ~on_land

        # Save the last floating location, for use in resuspension
        if np.sum(floating) > 0:
            self.elements.last_floating_lon[floating] = self.elements.lon[floating]
            self.elements.last_floating_lat[floating] = self.elements.lat[floating]
            logger.debug('Saved last floating positions')
                           
        # Tackle beaching and resuspension for particles on land
        N_beaching_particles = np.sum(on_land)
        if N_beaching_particles == 0:
            logger.debug('No elements hit coastline')
        else:            
            # TODO: Only do this for particles that were not beached this timestep
            logger.debug(f'Running beaching model for {N_beaching_particles} particles.')
            Tp = self.environment.sea_surface_wave_period_at_variance_spectral_density_maximum[on_land]
            dt = np.abs(self.time_step.total_seconds())
            Nw = np.int32(dt/Tp) # Number of waves in the timestep
            y = self.elements.height_on_beach[on_land]
            eta = self.environment.sea_surface_height[on_land]
            
            # Not sure if this is the way to do it but I need to somehow make sure that floating particles are at y=eta for the method to work 
            if np.any(y==0):
                y[y==0] = eta[y==0]
            if np.any(y < eta):
                y[y<eta] = eta[y<eta]
            
            
            # Compute sigma for the Rayleigh distribution
            H_s = self.environment.sea_surface_wave_significant_height[on_land]
            H_rms = 0.5 * np.sqrt(2) * H_s
            L = 9.81 * Tp**2 / (2*np.pi)
            alpha = self.environment.beach_angle[on_land]
            sigma = np.sqrt(H_rms*L)
            steep_beach = np.tan(alpha) > 0.1 
            if np.sum(steep_beach) > 0:
                sigma[steep_beach] *= 0.6 * np.tan(alpha)
            if np.sum(~steep_beach) > 0:
                sigma[~steep_beach] *= 0.05

            
            # TODO: Set p some other way
            p = 0.1
            if callable(p):
                # p is not a constant
                y = one_timestep_varying_p(y=y,
                                           p=p,
                                           Nw=Nw,
                                           t=self.time,
                                           scale=sigma,
                                           loc=eta)
            else:
                logger.debug("p is constant")
                y = self.one_timestep_constant_p(y=y, 
                                            p=p,
                                            Nw=Nw,
                                            t=self.time,
                                            scale=sigma,
                                            loc=eta)
                
            beached_mask = y > eta
            floating_mask = ~beached_mask

            self.elements.height_on_beach[on_land] = y 
            # Beached: stay beached
            if np.sum(beached_mask) > 0:
                logger.debug(f'{np.sum(beached_mask)} particles still beached')
                (self.elements.beached[on_land])[beached_mask] = 1

            # Floating: Put back to the last floating location they had
            if np.sum(floating_mask) > 0:
                logger.debug(f'{np.sum(floating_mask)} particles resuspended.')
                (self.elements.lon[on_land])[floating_mask] = (self.elements.last_floating_lon[on_land])[floating_mask]
                (self.elements.lat[on_land])[floating_mask] = (self.elements.last_floating_lat[on_land])[floating_mask]
                (self.elements.beached[on_land])[floating_mask] = 0
                (self.elements.height_on_beach[on_land])[floating_mask] = 0
                (self.elements.moving[on_land])[floating_mask] = 1 # Let resuspended particles move again 
            
            # TODO: Use a different variable than y






    ###### For beaching model
    def one_timestep_constant_p(self, y, p, Nw, t, residence_time=None, scale=1, loc=0):
        # Count the number of particles
        Np = len(y)

        # Array to count the number of waves remaining for each particle
        remaining_waves = Nw * np.ones(Np).astype(np.int64)
        
        # Repeat procedure as long as there are remainging waves for at least one particle
        while np.any(remaining_waves > 0):
            # Filter out the particles that we are dealing with in this iteration 
            active_mask = remaining_waves > 0
            active_particles = y[active_mask]
            remaining_waves_for_active_particles = remaining_waves[active_mask]
            loc_for_active_particles = loc[active_mask]
            scale_for_active_particles = scale[active_mask]

            # Separate the floating particles from the beached particles
            beached_mask = active_particles > (0 + loc_for_active_particles)
            floating_mask = ~beached_mask
            ### FLOATING PARTICLES
            if np.sum(floating_mask) > 0:
                remaining_waves_floating = remaining_waves_for_active_particles[floating_mask]
                floating_particles = active_particles[floating_mask]
                floating_loc = loc_for_active_particles[floating_mask]
                floating_scale = scale_for_active_particles[floating_mask]
                
                floating_particles = self.waves_on_floating_particles(Np=sum(floating_mask), 
                                                                      p=p, 
                                                                      Nw=remaining_waves_floating, 
                                                                      scale=floating_scale, 
                                                                      loc=floating_loc)
                
                # Done with all waves for these particles
                remaining_waves_for_active_particles[floating_mask] = 0
                active_particles[floating_mask] = floating_particles


            ### BEACHED PARTICLES 
            if np.sum(beached_mask) > 0:
                remaining_waves_beached = remaining_waves_for_active_particles[beached_mask]
                beached_particles = active_particles[beached_mask]
                beached_loc = loc_for_active_particles[beached_mask]
                beached_scale = scale_for_active_particles[beached_mask]

                # Probability of a wave being higher than the particle positions
                p_y = scipy.stats.rayleigh.sf(beached_particles, scale=beached_scale, loc=beached_loc)

                # Add a tolerance for p_y
                tol = 1e-5
                if np.any(p_y < tol):
                    p_y[p_y < tol] = tol
                # Number of waves it takes to get one that is high enough
                N_y = scipy.stats.geom.rvs(p_y) # Geometric distribution

                # The lower waves don't affect the particles so we can ignore them
                # Remove the lower waves from the number of remaining waves 
                #remaining_waves_beached -= N_y
                remaining_waves_for_active_particles[beached_mask] = remaining_waves_beached - N_y
                remaining_waves_beached = remaining_waves_beached - N_y
                # Check for particles that still have more waves remaining after we removed the lower ones 
                if np.any(remaining_waves_beached >= 0):
                    # For these particles, wave number N_y is high enough
                    # Now we need to figure out what happens when the wave hits

                    # Get the particles that still have remaining waves
                    still_more_waves_mask = remaining_waves_beached >= 0
                    n_remaining = np.sum(still_more_waves_mask) # Number of particles that still have remaning waves 
                    particles_affected_by_wave = beached_particles[still_more_waves_mask]
        
                    # Get p for the particles that are affected by the waves 
                    # and check if particles are pushed up or washed out 
                    r = np.random.random(n_remaining)
                    pushed_up = r < p # Mask for particles that are pushed up 
                    washed_out = ~pushed_up # Mask for particles that are washed out 


                    ### PUSHED UP PARTICLES
                    if np.sum(pushed_up) > 0:
                        # Get new position, then move on to next iteration
                        particles_affected_by_wave[pushed_up] = self.truncated_Rayleigh((p_y[still_more_waves_mask])[pushed_up], 
                                                                                        sigma=(beached_scale[still_more_waves_mask])[pushed_up], 
                                                                                        eta=(beached_loc[still_more_waves_mask])[pushed_up])

                    ### WASHED OUT PARTICLES
                    if np.sum(washed_out) > 0:
                        # Set to zero, then move on to next iteration 
                        particles_affected_by_wave[washed_out] = 0. + (beached_loc[still_more_waves_mask])[washed_out] 

                    # Put back in
                    beached_particles[still_more_waves_mask] = particles_affected_by_wave
                
                active_particles[beached_mask] = beached_particles
    
            # Update particle position and number of waves
            y[active_mask] = active_particles
            remaining_waves[active_mask] = np.array([max(w,0) for w in remaining_waves_for_active_particles])

        return y

    def waves_on_floating_particles(self, Np, p, Nw, scale=1, loc=0): # TODO: come up with a better name 
        """ Models the effect of Nw waves on Np particles that are initially floating, independent of each other. 
        That is, each of the Np particles is affected by Nw waves but the waves are different for each particle.
        For each wave, the particle has a probability p of being left on land at the height of the wave run up. 
        The probability p can be a constant or a function of position and time. 
        """
        # Initialise all particles to zero height above sea level (floating)
        y = np.zeros(Np) + loc 

        # Draw N from the betabinomial distribution
        N = scipy.stats.betabinom.rvs(n=Nw, a=p, b=1-p, size=Np)

        # Get heights for the particles on land
        nonzero = N > 0
        if np.sum(nonzero) > 0:
            y[nonzero] = self.get_RN(N[nonzero], scale=scale[nonzero], loc=loc[nonzero])

        # If N = 0 the particles are afloat (y=0) and we don't need to do anything

        return y
    

    def truncated_Rayleigh(self, p_r, sigma=1, eta=0):
        """Draws a random Rayleigh distributed number, under the constaint that is must be larger than a certain value r.
        p_r = sf(r), where sf(r) is the probability that a rayleigh distributed number is larger than r. 
        
        This is done by drawing a random uniformly distributed number in the interval [0, p_r] and converting it to a
        Rayleigh distributed number. 
        
        """
        u = np.random.uniform(0, p_r)
        return self.transform(u, scale=sigma, loc=eta)
    
    def get_RN(self, N, scale=1, loc=0):
        """ Returns the maximum of N Rayleigh distributed numbers.
        
        This is done by finding the minimum of a uniformly distributed number between zero and one, 
        and converting it to a Rayleigh distributed number.
        """
        return self.transform(self.get_U_min(N), scale=scale, loc=loc)
    
    def transform(self, U, scale=1,  loc=0):
        """ Converts a uniformly distributed number to a Rayleigh distributed number with scale sigma. """
        return scale * np.sqrt(-2*np.log(U)) + loc
    

    def get_U_min(self, N):
        """ Returns the minimum of N uniformly distributed numbers in the interval [0, 1] """
        U_min = scipy.stats.beta.rvs(a=1, b=N)
        return U_min