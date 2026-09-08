import { useEffect, useState } from "react";
import { fetchDependencyHealth } from "../api/client";
import type { ServiceHealth } from "../types";

export default function ServiceHealthList() {
  const [services, setServices] = useState<ServiceHealth[]>([]);

  useEffect(() => {
    let active = true;

    function poll() {
      fetchDependencyHealth()
        .then((r) => {
          if (active) setServices(r.services);
        })
        .catch(() => {
          if (active) setServices([]);
        });
    }

    poll();
    const id = setInterval(poll, 30_000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  if (services.length === 0) return null;

  return (
    <div className="service-health">
      <span className="eyebrow service-health__title">Services</span>
      <ul className="service-health__list">
        {services.map((s) => (
          <li key={s.name} className="service-health__item" title={s.status === "up" ? "Reachable" : "Unreachable"}>
            <span className={`service-health__dot service-health__dot--${s.status}`} aria-hidden="true" />
            {s.label}
          </li>
        ))}
      </ul>
    </div>
  );
}
