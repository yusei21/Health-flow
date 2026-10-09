export interface PatientRecord {
  display_name: string;
  age: number | null;
  allergies: string[];
  conditions: { name: string }[];
  active_medications: { name: string }[];
  previous_encounters: {
    occurred_on: string;
    service: string;
    summary: string;
  }[];
}
export interface Capabilities {
  govbr_available: boolean;
  demo_available: boolean;
  audio_available: boolean;
  record_source: string;
}
export interface SUSAnswer {
  status: "not_verified";
  answer: string;
  sources: { title: string; url: string }[];
}
