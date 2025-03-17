#include "ERF_Morrison.H"
#include "ERF_IndexDefines.H"
#include "ERF_TileNoZ.H"
#include "ERF_EOS.H"
 
using namespace amrex;

/**
 * Split cloud components according to saturation pressures; source theta from latent heat.
 */
void
Morrison::Cloud (const SolverChoice& /*sc*/)
{
    BL_PROFILE("Morrison::Cloud()");

    constexpr Real tbgmin = 253.15;
    constexpr Real tbgmax = 273.15;
    constexpr Real an = 1.0/(tbgmax - tbgmin);
    constexpr Real bn = tbgmin*an;

    Real fac_cond = m_fac_cond;
    Real fac_sub  = m_fac_sub;
    Real fac_fus  = m_fac_fus;
    Real rdOcp    = m_rdOcp;

    for (MFIter mfi(*(mic_fab_vars[MicVar_Morr::tabs]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto  qt_array = mic_fab_vars[MicVar_Morr::qt]->array(mfi);
        auto  qn_array = mic_fab_vars[MicVar_Morr::qn]->array(mfi);
        auto  qv_array = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto qcl_array = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto qci_array = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto qg_array = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto qr_array = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto  rho_array = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto  tabs_array = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto theta_array = mic_fab_vars[MicVar_Morr::theta]->array(mfi);
        auto  pres_array = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
        auto  nc_array   = mic_fab_vars[MicVar_Morr::nc]->array(mfi);
        auto  nr_array   = mic_fab_vars[MicVar_Morr::nr]->array(mfi);
        auto  ni_array   = mic_fab_vars[MicVar_Morr::ni]->array(mfi);
        auto  ns_array   = mic_fab_vars[MicVar_Morr::ns]->array(mfi);
        auto  ng_array   = mic_fab_vars[MicVar_Morr::ng]->array(mfi);
        auto  w_array    = mic_fab_vars[MicVar_Morr::omega]->array(mfi);

        const auto& box3d = mfi.tilebox();

        ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Cloud phase adjustments and saturation
            Real temp = tabs_array(i,j,k);
            Real pres = pres_array(i,j,k);
            Real rho = rho_array(i,j,k);
            Real qv = qv_array(i,j,k);
            Real qc = qcl_array(i,j,k);
            Real qi = qci_array(i,j,k);
            Real qg = qg_array(i,j,k);
            Real qr = qr_array(i,j,k);
            Real nc = nc_array(i,j,k);
            Real nr = nr_array(i,j,k);
            Real ni = ni_array(i,j,k);
            Real ns = ns_array(i,j,k);
            Real ng = ng_array(i,j,k);
            Real w_local = w_array(i,j,k);

            // CCN Activation
            if (temp > 273.15 && qc < 0.05e-3) {
                Real w_eff = w_local;
                if (m_isub == 0) w_eff = std::sqrt(w_local*w_local + 0.5*0.5);
                if (w_eff > 0.01) {
                    Real supersat = calculateSupersaturation(w_eff, temp, pres, qv);
                    Real nact = 0.0;

                    if (m_iact == 1) {
                        nact = m_c1 * std::pow(supersat*100.0, m_k1) * 1.0e6 / rho;
                    } else if (m_iact == 2) {
                        Real alpha = std::pow(2.0*m_mw*0.0761/(m_rhow*m_r_v*temp), 1.5);
                        Real gamma = m_r_v*temp*m_rhow/(m_mw*0.0761);
                        Real psi = 2.0/3.0 * std::sqrt(alpha/gamma);
                        Real eta1 = std::pow(supersat/(m_f11*m_nanew1), 1.0/m_f21);
                        Real eta2 = std::pow(supersat/(m_f12*m_nanew2), 1.0/m_f22);
                        Real smax = supersat;
                        Real uu1 = 2.0*std::log(m_rm1/m_bact) / (3.0*std::sqrt(2.0)*std::log(m_sig1));
                        Real uu2 = 2.0*std::log(m_rm2/m_bact) / (3.0*std::sqrt(2.0)*std::log(m_sig2));
                        Real n1 = 0.5*m_nanew1*(1.0 - ErrorFunction(uu1));
                        Real n2 = 0.5*m_nanew2*(1.0 - ErrorFunction(uu2));
                        nact = (n1 + n2) / rho;
                    }

                    nact = std::min(nact, 1.0e10);
                    Real nccn_max = 1000.0e6 / rho;
                    nact = std::min(nact, nccn_max);
                    if (nc < nact) {
                        Real pccn = (nact - nc) / dt;
                        nc_array(i,j,k) += pccn * dt;
                    }
                }
            }

            // Homogeneous Freezing
            if (temp <= 233.15 && qc >= m_qsmall) {
                qi += qc;
                Real xxlv = 3.1484e6 - 2370.0 * temp;
                Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;
                Real xlf = xxls - xxlv;
                Real cpm = m_cp * (1.0 + 0.887 * qv);
                tabs_array(i,j,k) += qc * xlf / cpm;
                ni_array(i,j,k) += nc_array(i,j,k);
                qc = 0.0;
                nc_array(i,j,k) = 0.0;
            }

            if (temp <= 233.15 && qr >= m_qsmall) {
                qg += qr;
                Real xxlv = 3.1484e6 - 2370.0 * temp;
                Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;
                Real xlf = xxls - xxlv;
                Real cpm = m_cp * (1.0 + 0.887 * qv);
                tabs_array(i,j,k) += qr * xlf / cpm;
                ng_array(i,j,k) += nr_array(i,j,k);
                qr = 0.0;
                nr_array(i,j,k) = 0.0;
            }

            // Saturation Adjustment
            if (qc < m_qsmall && qi < m_qsmall && qv < m_qsmall) return;

            Real T = temp;
            Real evs = calc_saturation_vapor_pressure(T, 0);
            Real eis = calc_saturation_vapor_pressure(T, 1);
            evs = std::min(0.99*pres, evs);
            eis = std::min(0.99*pres, eis);
            if (eis > evs) eis = evs;

            Real qvs = m_ep_2 * evs / (pres - evs);
            Real qvi = m_ep_2 * eis / (pres - eis);
            Real qsat = (an*T - bn) * qvs + (1.0 - (an*T - bn)) * qvi;

            if (qv > qsat) {
                // Newton Iteration
                int niter = 0;
                Real dtabs = 1.0;
                Real tol = 1.0e-4;

                while (std::abs(dtabs) > tol && niter < 20) {
                    Real omn = (T >= tbgmax) ? 1.0 : ((T <= tbgmin) ? 0.0 : (an*T - bn));
                    Real domn = (T >= tbgmax || T <= tbgmin) ? 0.0 : an;

                    evs = calc_saturation_vapor_pressure(T, 0);
                    eis = calc_saturation_vapor_pressure(T, 1);
                    evs = std::min(0.99*pres, evs);
                    eis = std::min(0.99*pres, eis);
                    if (eis > evs) eis = evs;

                    qvs = m_ep_2 * evs / (pres - evs);
                    qvi = m_ep_2 * eis / (pres - eis);

                    Real dum = m_Rv * T * T;
                    Real dqsdt = (3.1484e6 - 2370.0 * T) * qvs / dum;
                    Real dqsidt = (3.15e6 - 2370.0 * T + 0.3337e6) * qvi / dum;

                    Real qsat = omn * qvs + (1.0 - omn) * qvi;
                    Real dqsat = omn * dqsdt + (1.0 - omn) * dqsidt + domn * (qvs - qvi);

                    Real lsterms = omn * m_fac_cond + (1.0 - omn) * m_fac_sub;
                    Real dlsterms = domn * (m_fac_cond - m_fac_sub);

                    Real f = -T + temp + lsterms * (qv - qsat);
                    Real df = -1.0 + dlsterms * (qv - qsat) - lsterms * dqsat;

                    dtabs = -f / df;
                    T += dtabs;
                    niter++;
                }

                tabs_array(i,j,k) = T;
                theta_array(i,j,k) = getThgivenPandT(T, 100.0*pres, rdOcp);
                pres_array(i,j,k) *= 0.01;
            } else {
                Real delta_qv = qc + qi;
                Real delta_qc = qc;
                Real delta_qi = qi;
                qv += delta_qv;
                qc = 0.0;
                qi = 0.0;
                qn_array(i,j,k) = 0.0;
                qt_array(i,j,k) = qv;

//                tabs_array(i,j,k) -= fac_cond * delta_qc + fac_sub * delta_qi;
//                theta_array(i,j,k) = getThgivenPandT(tabs_array(i,j,k), 100.0*pres, rdOcp);
/*
                // Calculate the deficit in water vapor
                Real delta_qv = qv - qsat; // Negative value indicates deficit

                // Determine phase partitioning based on temperature
                Real omn = (T >= tbgmax) ? 1.0 : ((T <= tbgmin) ? 0.0 : (an * T - bn));

                // Partition deficit between liquid and ice
                // Ensure delta_qc and delta_qi are positive when there's a deficit
                Real delta_qc = std::min(qc, -delta_qv * omn);    // Amount of liquid water to condense
                Real delta_qi = std::min(qi, -delta_qv * (1.0 - omn)); // Amount of ice to form

                // Update mixing ratios
                qv_array(i,j,k) = qsat; // Set vapor to saturation
                qcl_array(i,j,k) += delta_qc; // Add condensed liquid water
                qci_array(i,j,k) += delta_qi; // Add formed ice
*/
                // Calculate heat capacity including water vapor
                Real cpm = m_cp * (1.0 + 0.887 * qv_array(i,j,k));

                // Apply latent heating (positive when condensing/cooling)
                tabs_array(i,j,k) += (delta_qc * m_fac_cond + delta_qi * m_fac_sub) / cpm;

                // Update potential temperature
                theta_array(i,j,k) = getThgivenPandT(tabs_array(i,j,k), 100.0 * pres, rdOcp);

                evs = calc_saturation_vapor_pressure(tabs_array(i,j,k), 0);
                eis = calc_saturation_vapor_pressure(tabs_array(i,j,k), 1);
                evs = std::min(0.99*pres, evs);
                eis = std::min(0.99*pres, eis);
                if (eis > evs) eis = evs;

                qvs = m_ep_2 * evs / (pres - evs);
                qvi = m_ep_2 * eis / (pres - eis);
                qsat = (an*tabs_array(i,j,k) - bn) * qvs + (1.0 - (an*tabs_array(i,j,k) - bn)) * qvi;

                if (qt_array(i,j,k) > qsat) {
                    // Repeat the Newton-Raphson iteration
                    Real T_new = tabs_array(i,j,k);
                    Real dtabs_new = 1.0;
                    int niter_new = 0;
                    Real tol_new = 1.0e-4;

                    while (std::abs(dtabs_new) > tol_new && niter_new < 20) {
                        Real omn_new = (T_new >= tbgmax) ? 1.0 : ((T_new <= tbgmin) ? 0.0 : (an*T_new - bn));
                        Real domn_new = (T_new >= tbgmax || T_new <= tbgmin) ? 0.0 : an;

                        evs = calc_saturation_vapor_pressure(T_new, 0);
                        eis = calc_saturation_vapor_pressure(T_new, 1);
                        evs = std::min(0.99*pres, evs);
                        eis = std::min(0.99*pres, eis);
                        if (eis > evs) eis = evs;

                        qvs = m_ep_2 * evs / (pres - evs);
                        qvi = m_ep_2 * eis / (pres - eis);

                        Real dum = m_Rv * T_new * T_new;
                        Real dqsdt = (3.1484e6 - 2370.0 * T_new) * qvs / dum;
                        Real dqsidt = (3.15e6 - 2370.0 * T_new + 0.3337e6) * qvi / dum;

                        Real qsat_new = omn_new * qvs + (1.0 - omn_new) * qvi;
                        Real dqsat_new = omn_new * dqsdt + (1.0 - omn_new) * dqsidt + domn_new * (qvs - qvi);

                        Real lsterms_new = omn_new * m_fac_cond + (1.0 - omn_new) * m_fac_sub;
                        Real dlsterms_new = domn_new * (m_fac_cond - m_fac_sub);

                        Real f_new = -T_new + tabs_array(i,j,k) + lsterms_new * (qv - qsat_new);
                        Real df_new = -1.0 + dlsterms_new * (qv - qsat_new) - lsterms_new * dqsat_new;

                        dtabs_new = -f_new / df_new;
                        T_new += dtabs_new;
                        niter_new++;
                    }

                    tabs_array(i,j,k) = T_new;
                    theta_array(i,j,k) = getThgivenPandT(T_new, 100.0*pres, rdOcp);
                }
            }

            // Heterogeneous Freezing
            if (temp < 269.15 && temp > 233.15 && qc >= m_qsmall) {
                Real n_contact = std::exp(-2.80 + 0.262 * (273.15 - temp)) * 1000.0;
                Real dum = 7.37 * temp / (288.0 * 10.0 * pres) / 100.0;
                Real dap = m_cons37 * temp * (1.0 + dum / m_rin) / m_mu;
                Real dum1 = pres / (287.15 * temp);

                Real pgam = 0.0005714 * (nc * rho / 1.0e6 * dum1) + 0.2714;
                pgam = 1.0 / (pgam * pgam) - 1.0;
                pgam = std::max(pgam, 2.0);
                pgam = std::min(pgam, 10.0);

                Real lamc = std::pow(m_cons26 * nc * gamma_function(pgam + 4.0) / (qc * gamma_function(pgam + 1.0)), 1.0/3.0);
                Real cdist = nc / gamma_function(pgam + 1.0);

                Real mnuccc = m_cons38 * dap * n_contact * std::exp(std::log(cdist) + std::log(gamma_function(pgam + 5.0)) - 4.0 * std::log(lamc));
                Real nnuccc = 2.0 * M_PI * dap * n_contact * cdist * gamma_function(pgam + 2.0) / lamc;

                Real imm_rate = std::exp(m_aimm * (273.15 - temp)) - 1.0;
                Real mnuccc_imm = m_cons39 * std::exp(std::log(cdist) + std::log(gamma_function(7.0 + pgam)) - 6.0 * std::log(lamc)) * imm_rate;
                Real nnuccc_imm = m_cons40 * std::exp(std::log(cdist) + std::log(gamma_function(pgam + 4.0)) - 3.0 * std::log(lamc)) * imm_rate;

                Real mnuc_total = mnuccc + mnuccc_imm;
                Real nnuc_total = nnuccc + nnuccc_imm;

                Real max_freeze_qc = qc / dt;
                Real max_freeze_nc = nc / dt;

                Real mnuc_limited = std::min(mnuc_total, max_freeze_qc);
                Real nnuc_limited = std::min(nnuc_total, max_freeze_nc);

                qi += mnuc_limited * dt;
                ni += nnuc_limited * dt;
                qc -= mnuc_limited * dt;
                nc -= nnuc_limited * dt;

                Real xxlv = 3.1484e6 - 2370.0 * temp;
                Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;
                Real xlf = xxls - xxlv;
                Real cpm = m_cp * (1.0 + 0.887 * qv);
                tabs_array(i,j,k) += mnuc_limited * xlf / cpm * dt;
            }
#if 0
            // Primary Ice Nucleation
            if (temp < 273.15) {
                Real evs = calc_saturation_vapor_pressure(temp, 0);
                Real eis = calc_saturation_vapor_pressure(temp, 1);
                evs = std::min(0.99*pres, evs);
                eis = std::min(0.99*pres, eis);
                if (eis > evs) eis = evs;

                Real qvs = m_ep_2 * evs / (pres - evs);
                Real qvi = m_ep_2 * eis / (pres - eis);
                Real qvqvs = qv / qvs;
                Real qvqvsi = qv / qvi;

                if (m_inuc_type == 0 && qvqvsi >= 1.0 && temp <= 265.15) {
                    Real kc2 = 0.005 * std::exp(0.304 * (t_freeze - temp)) * 1000.0;
                    kc2 = std::min(kc2, 500.0e3);
                    kc2 /= rho;

                    if (kc2 > ni + ns + ng) {
                        Real nnuccd = (kc2 - (ni + ns + ng)) / dt;
                        Real mnuccd = nnuccd * m_mi0;
                        qi += mnuccd * dt;
                        ni += nnuccd * dt;
                        tabs_array(i,j,k) += mnuccd * m_fac_sub * dt / (m_cp * (1.0 + 0.887 * qv));
                    }
                } else if (m_inuc_type == 1 && qvqvsi > 1.0) {
                    Real kc2 = 0.16 * 1000.0;
                    kc2 /= rho;

                    if (kc2 > ni + ns + ng) {
                        Real nnuccd = (kc2 - (ni + ns + ng)) / dt;
                        Real mnuccd = nnuccd * m_mi0;
                        qi += mnuccd * dt;
                        ni += nnuccd * dt;
                        tabs_array(i,j,k) += mnuccd * m_fac_sub * dt / (m_cp * (1.0 + 0.887 * qv));
                    }
                }
            }

            // Ice-Snow Categorization
            if (temp < t_freeze && qi >= m_qsmall) {
                Real lami = std::pow(m_cons12 * ni / qi, 1.0/m_di);
                if (lami >= 1.0e-10 && 1.0/lami >= 2.0*m_dcs) {
                    qs += qi;
                    ns += ni;
                    qi = 0.0;
                    ni = 0.0;
                }
            }
#endif
        });
    }
}
