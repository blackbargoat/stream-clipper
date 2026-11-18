-- Clipora settings: connected accounts (mirrored from Buffer), auto-post per account, auto-approve.
create table if not exists public.clip_settings (key text primary key, value jsonb not null, updated_at timestamptz default now());
alter table public.clip_settings enable row level security;
insert into public.clip_settings (key, value) values
  ('channels', '[{"id":"6af749df62b70b94b01e02ec","name":"Joe Bart Unhinged","service":"youtube","avatar":"https://yt3.ggpht.com/4OAg2ySNw7nkSLAg4UlQlg2sJwlvKAzIS6X7tW1yyM0WyfQJConEzDj_yBszKEIITWo8B-s0YAA=s88-c-k-c0x00ffffff-no-rj","autopost":false}]'),
  ('auto_approve', 'false'),
  ('min_auto_score', '4.0')
on conflict (key) do nothing;
